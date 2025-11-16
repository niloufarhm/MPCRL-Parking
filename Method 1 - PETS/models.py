
import torch
from torch import nn


class ResidualDynamics(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, s, a):
        net_input = torch.cat([s, a], dim=-1)

        delta_s = self.net(net_input)
        s_next = s + delta_s

        sin_next = s_next[:, 4:5]
        cos_next = s_next[:, 5:6]
        norm = torch.sqrt(sin_next ** 2 + cos_next ** 2 + 1e-6)

        s_next_normalized = s_next.clone()
        s_next_normalized[:, 4:5] = sin_next / norm
        s_next_normalized[:, 5:6] = cos_next / norm

        return s_next_normalized