#!/usr/bin/env python3
"""
rollout_view.py — visualize a trained TD-MPC2 agent (.pt) interactively.

Usage:
  python rollout_view.py --ckpt checkpoints/tdmpc2_step_50000.pt --planner actor --episodes 5
  python rollout_view.py --latest --planner mpc --episodes 3
"""

import os, argparse, torch, numpy as np, gymnasium as gym, highway_env
from agent import TD_MPC2_Agent


# ---------- Constants ----------
CKPT_DIR = "checkpoints"


# ---------- Environment ----------
def make_env(render=True, seed=None):
    render_mode = "human" if render else "rgb_array"
    env = gym.make("parking-v0", render_mode=render_mode)
    env.unwrapped.configure({
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": True,
        },
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offscreen_rendering": not render,
    })
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    return env


def flatten_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)


# ---------- Agent helpers ----------
@torch.no_grad()
def policy_action(agent: TD_MPC2_Agent, obs, sample=False):
    device = agent.device
    z = agent.wm.encode(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
    a, dist = agent.actor(z)
    if not sample:
        a = torch.tanh(dist.mean)
    return a.squeeze(0).detach().cpu().numpy()


@torch.no_grad()
def plan_cem_light(agent: TD_MPC2_Agent, obs, horizon=6, pop=64, iters=3, elite_frac=0.1, discount=0.99):
    device = agent.device
    act_dim = agent.actor.net[-1].out_features // 2
    z0 = agent.wm.encode(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))

    elites = max(1, int(pop * elite_frac))
    mean = torch.zeros(horizon, act_dim, device=device)
    std = torch.ones_like(mean) * 0.5

    for _ in range(iters):
        actions = torch.normal(mean.expand(pop, -1, -1), std.expand(pop, -1, -1))
        returns = []
        for i in range(pop):
            z = z0.clone()
            G, gamma = 0.0, 1.0
            for a in actions[i]:
                z, r = agent.wm.predict(z, a.unsqueeze(0))
                G += gamma * r.squeeze()
                gamma *= discount
            G += gamma * agent.val(z).squeeze()
            returns.append(G)
        returns = torch.stack(returns)
        top_idx = torch.topk(returns, elites).indices
        elite_actions = actions[top_idx]
        mean, std = elite_actions.mean(0), elite_actions.std(0) + 1e-4

    a0 = mean[0].clamp(-1, 1)
    return a0.detach().cpu().numpy()


# ---------- Checkpoint helpers ----------
def find_latest_ckpt():
    ckpts = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt") and f.startswith("tdmpc2_step_")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
    return os.path.join(CKPT_DIR, ckpts[-1])


def load_ckpt(agent: TD_MPC2_Agent, path: str):
    ckpt = torch.load(path, map_location="cpu")
    agent.wm.load_state_dict(ckpt["wm"])
    agent.val.load_state_dict(ckpt["val"])
    agent.actor.load_state_dict(ckpt["actor"])
    print(f"[loaded] {path}")


# ---------- Rollout ----------
def rollout(agent, env, planner="actor", max_steps=400):
    ob, _ = env.reset()
    ob = flatten_obs(ob)
    total_reward = 0.0
    done, t = False, 0

    while not done and t < max_steps:
        if planner == "mpc":
            a = plan_cem_light(agent, ob)
        else:
            a = policy_action(agent, ob, sample=False)
        a = np.clip(a, -1.0, 1.0)
        ob2, r, term, trunc, _ = env.step(a)
        total_reward += r
        done = term or trunc
        ob = flatten_obs(ob2)
        t += 1
    return total_reward, t


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=str, help="Path to specific .pt checkpoint")
    src.add_argument("--latest", action="store_true", help="Use latest checkpoint")
    parser.add_argument("--planner", choices=["mpc", "actor"], default="actor")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if args.latest:
        ckpt_path = find_latest_ckpt()
        if ckpt_path is None:
            raise FileNotFoundError("No checkpoints found in 'checkpoints/'")

    env = make_env(render=True)
    agent = TD_MPC2_Agent(obs_dim=12, act_dim=2, device=args.device)
    load_ckpt(agent, ckpt_path)

    all_returns = []
    for ep in range(args.episodes):
        print(f"\n[Episode {ep+1}/{args.episodes}] using planner={args.planner}")
        ret, steps = rollout(agent, env, planner=args.planner, max_steps=args.max_steps)
        print(f"  Return: {ret:.2f} | Steps: {steps}")
        all_returns.append(ret)

    print(f"\n[Summary] {args.episodes} episodes → avg return = {np.mean(all_returns):.2f}")
    env.close()


if __name__ == "__main__":
    main()
