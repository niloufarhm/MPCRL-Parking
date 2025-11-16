import torch, torch.nn as nn, torch.optim as optim
import numpy as np

import torch.nn.functional as F

def contrastive_loss(z_pred, z_target, temperature=0.1):
    """
    Simple InfoNCE-style contrastive loss between predicted and true next latents.
    """
    z_pred = F.normalize(z_pred, dim=-1)
    z_target = F.normalize(z_target, dim=-1)

    logits = z_pred @ z_target.T / temperature  # (B, B)
    labels = torch.arange(len(z_pred), device=z_pred.device)
    return F.cross_entropy(logits, labels)

# ---------- small MLP factory ----------
def mlp(in_dim, out_dim, hid=256, layers=3):
    net = []
    for _ in range(layers):
        net += [nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.SiLU(inplace=False)]
        in_dim = hid
    net += [nn.Linear(in_dim, out_dim)]
    return nn.Sequential(*net)


# ---------- World Model (latent dynamics) ----------
class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim=64, hid=256):
        super().__init__()
        self.encoder = mlp(obs_dim, latent_dim, hid)
        self.transition = mlp(latent_dim + act_dim, latent_dim, hid)
        self.reward = mlp(latent_dim + act_dim, 1, hid)

    def encode(self, obs):
        return self.encoder(obs)

    def predict(self, z, a):
        za = torch.cat([z, a], -1)
        z_next = self.transition(za)
        r_pred = self.reward(za)
        return z_next, r_pred


# ---------- Value Function ----------
class ValueNet(nn.Module):
    def __init__(self, latent_dim, hid=256):
        super().__init__()
        self.v = mlp(latent_dim, 1, hid)

    def forward(self, z):
        return self.v(z)


# ---------- Actor (learned policy for warm-start) ----------
class Actor(nn.Module):
    def __init__(self, latent_dim, act_dim, hid=256):
        super().__init__()
        self.net = mlp(latent_dim, 2 * act_dim, hid)

    def forward(self, z):
        mu_logstd = self.net(z)
        mu, logstd = mu_logstd.chunk(2, -1)
        std = torch.exp(torch.clamp(logstd, -5, 1))
        dist = torch.distributions.Normal(mu, std)
        a = dist.rsample()
        return torch.tanh(a), dist


# ---------- CEM Planning (Model Predictive Control) ----------
@torch.no_grad()
def cem_plan(model, value_fn, z0, act_dim, horizon=5, pop=64, elite_frac=0.1, iters=3):
    elites = int(pop * elite_frac)
    mean = torch.zeros(horizon, act_dim, device=z0.device)
    std = torch.ones_like(mean) * 0.5
    for _ in range(iters):
        actions = torch.normal(mean.expand(pop, -1, -1), std.expand(pop, -1, -1))  # (pop,H,act)
        returns = []
        for i in range(pop):
            z, G, discount = z0.clone(), 0, 1
            for a in actions[i]:
                z, r = model.predict(z, a.unsqueeze(0))
                G += discount * r.squeeze()
                discount *= 0.99
            G += discount * value_fn(z).squeeze()
            returns.append(G)
        returns = torch.stack(returns)
        top = torch.topk(returns, elites).indices
        elite_actions = actions[top]
        mean, std = elite_actions.mean(0), elite_actions.std(0) + 1e-4
    return mean[0].clamp(-1, 1)


# ---------- TD-MPC2 Agent ----------
class TD_MPC2_Agent:
    def __init__(self, obs_dim, act_dim, device="cpu"):
        self.device = torch.device(device)
        self.wm = WorldModel(obs_dim, act_dim).to(self.device)
        self.val = ValueNet(64).to(self.device)
        self.actor = Actor(64, act_dim).to(self.device)
        self.opt = optim.Adam(list(self.wm.parameters()) + list(self.val.parameters()), lr=3e-4)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.gamma = 0.99

    def plan(self, obs):
        z0 = self.wm.encode(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        return cem_plan(self.wm, self.val, z0, act_dim=self.actor.net[-1].out_features // 2).cpu().numpy()

    # def update(self, batch):
    #     o, a, r, d, o2 = [x.to(self.device).float() for x in batch]
    #     z, z2 = self.wm.encode(o), self.wm.encode(o2)
    #     z_pred, r_pred = self.wm.predict(z, a)
    #     model_loss = (z_pred - z2.detach()).pow(2).mean() + (r_pred - r).pow(2).mean()
    #
    #     with torch.no_grad():
    #         v_target = r + self.gamma * (1 - d) * self.val(z2)
    #     v_loss = (self.val(z) - v_target).pow(2).mean()
    #
    #     loss = model_loss + v_loss
    #     self.opt.zero_grad()
    #     loss.backward()
    #     self.opt.step()
    #     return loss.item()

    # adding the actor training:
    def update(self, batch):
        o, a, r, d, o2 = [x.to(self.device).float() for x in batch]

        # --- Encode observations ---
        #z, z2 = self.wm.encode(o), self.wm.encode(o2)
        #adding augjenrtation
        noise_std = 0.01
        o_aug = o + noise_std * torch.randn_like(o)
        o2_aug = o2 + noise_std * torch.randn_like(o2)

        z, z2 = self.wm.encode(o_aug), self.wm.encode(o2_aug)

        # --- World model predictions ---
        z_pred, r_pred = self.wm.predict(z, a)

        # --- Contrastive representation loss ---
        contrast_loss = contrastive_loss(z_pred, z2.detach())

        # --- World model loss (latent + reward prediction) ---
        #model_loss = (z_pred - z2.detach()).pow(2).mean() + (r_pred - r).pow(2).mean()
        model_loss = (z_pred - z2.detach()).pow(2).mean() + (r_pred - r).pow(2).mean() + 0.1 * contrast_loss

        # --- Value loss (1-step TD target) ---
        with torch.no_grad():
            v_next = self.val(z2)
            v_target = r + self.gamma * (1 - d) * (v_next - 0.01 * v_next.pow(2))
        v_loss = (self.val(z) - v_target).pow(2).mean()

        # --- Optimize world model + value ---
        total_loss = model_loss + v_loss
        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()
        self.opt.step()

        # --- Imagination rollouts ---
        horizon_imagine = 3
        z_im = z.detach()
        returns_im = torch.zeros_like(r)
        for t in range(horizon_imagine):
            a_im, _ = self.actor(z_im)
            z_im, r_im = self.wm.predict(z_im, a_im)
            returns_im += (self.gamma ** t) * r_im
        v_im_loss = (self.val(z.detach()) - returns_im.detach()).pow(2).mean()

        self.opt.zero_grad(set_to_none=True)
        v_im_loss.backward()
        self.opt.step()

        # --- Actor learning (imitation + value maximization) ---
        with torch.no_grad():
            # pick a random latent from the batch and plan an MPC teacher action
            idx = torch.randint(0, z.size(0), (1,))
            a_mpc = cem_plan(self.wm, self.val, z[idx].detach(), act_dim=a.shape[-1])

        # detach everything for independent gradient flow
        z_actor = z.detach()
        a_mpc = a_mpc.detach()

        a_actor, _ = self.actor(z_actor)
        imit_loss = (a_actor - a_mpc).pow(2).mean()
        value_loss = -self.val(z_actor.detach()).mean()
        actor_loss = imit_loss + 0.1 * value_loss

        # optimize actor separately
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # --- Return all loss terms for logging ---
        return {
            "total_loss": total_loss.item(),
            "model_loss": model_loss.item(),
            "v_loss": v_loss.item(),
            "actor_loss": actor_loss.item()
        }

    # --- Checkpoint helpers ---
    def state_dict(self):
        """Collect all model and optimizer weights for checkpointing."""
        return {
            "wm": self.wm.state_dict(),
            "val": self.val.state_dict(),
            "actor": self.actor.state_dict(),
            "opt": self.opt.state_dict(),
            "gamma": self.gamma,
        }

    def load_state_dict(self, state_dict):
        """Reload all model and optimizer weights from checkpoint."""
        self.wm.load_state_dict(state_dict["wm"])
        self.val.load_state_dict(state_dict["val"])
        self.actor.load_state_dict(state_dict["actor"])
        if "opt" in state_dict:
            self.opt.load_state_dict(state_dict["opt"])
        self.gamma = state_dict.get("gamma", self.gamma)
