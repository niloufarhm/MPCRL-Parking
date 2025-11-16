import numpy as np
import torch


def angular_cost(s_sin, s_cos, g_sin, g_cos):
    return 1.0 - (s_sin * g_sin + s_cos * g_cos)


def evaluate_sequences(fwd, s0, sg, sequences, device,
                       pos_w=3.0, head_w=5.0,
                       action_pen=0.005, smooth_pen=0.25,
                       final_pos_w=10.0, final_head_w=20.0, final_speed_w=20.0):

    pop, H, act_dim = sequences.shape
    s = s0.unsqueeze(0).repeat(pop, 1)
    sg_t = sg.unsqueeze(0).repeat(pop, 1)

    seq = torch.tensor(sequences, dtype=torch.float32, device=device)
    costs = torch.zeros(pop, device=device)
    a_prev = torch.zeros(pop, act_dim, device=device)

    with torch.no_grad():
        for t in range(H):
            a_t = seq[:, t, :]
            s = fwd(s, a_t)

            pos_cost = torch.norm(s[:, :2] - sg_t[:, :2], dim=-1)
            head_cost = angular_cost(s[:, 4], s[:, 5], sg_t[:, 4], sg_t[:, 5])

            costs += pos_w * pos_cost + head_w * head_cost
            costs += action_pen * torch.norm(a_t, dim=-1)
            costs += smooth_pen * torch.norm(a_t - a_prev, dim=-1)

            a_prev = a_t

        costs += final_pos_w * torch.norm(s[:, :2] - sg_t[:, :2], dim=-1)
        costs += final_head_w * angular_cost(s[:, 4], s[:, 5], sg_t[:, 4], sg_t[:, 5])
        final_speed_cost = torch.norm(s[:, 2:4], dim=-1)
        costs += final_speed_w * final_speed_cost

    return costs.cpu().numpy()


class CEMPlannerState:

    def __init__(self, forward_model, action_low, action_high, action_dim):
        self.fwd = forward_model
        self.action_dim = action_dim
        self.low = action_low
        self.high = action_high
        self.prev_solution = None

    def plan(self, s0, sg, horizon=24, iters=8, pop=1024, elite_frac=0.1, device='cpu'):
        s0_t = torch.tensor(s0, dtype=torch.float32, device=device)
        sg_t = torch.tensor(sg, dtype=torch.float32, device=device)

        if self.prev_solution is not None and self.prev_solution.shape[0] == horizon:
            mean = np.roll(self.prev_solution, -1, axis=0)
            mean[-1] = 0
        else:
            mean = np.zeros((horizon, self.action_dim), dtype=np.float32)

        std = np.ones_like(mean) * 0.5
        n_elite = max(1, int(pop * elite_frac))

        for iteration in range(iters):
            noise_scale = max(0.1, 1.0 - iteration / iters)  # Annealing
            samples = np.random.randn(pop, horizon, self.action_dim).astype(np.float32) * std * noise_scale + mean
            samples = np.clip(samples, self.low, self.high)

            costs = evaluate_sequences(self.fwd, s0_t, sg_t, samples, device)

            elite_idx = costs.argsort()[:n_elite]
            elites = samples[elite_idx]

            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-6
        self.prev_solution = mean
        return mean[0]

def success_state(s, sg, pos_thresh=0.15, head_thresh=0.1, speed_thresh=0.2):
    pos_err = np.linalg.norm(s[:2] - sg[:2])

    theta_s = np.arctan2(s[4], s[5])
    theta_g = np.arctan2(sg[4], sg[5])
    ang_err = np.abs(np.arctan2(np.sin(theta_s - theta_g), np.cos(theta_s - theta_g)))

    speed = np.linalg.norm(s[2:4])

    return (pos_err < pos_thresh) and (ang_err < head_thresh) and (speed < speed_thresh)