# replay_buffer.py
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=int(1e6)):
        self.ptr, self.size, self.full = 0, size, False
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.rew = np.zeros((size, 1), np.float32)
        self.done = np.zeros((size, 1), np.float32)

    def add(self, o, a, r, d, o2):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.done[i], self.next[i] = o, a, r, d, o2
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, batch):
        maxidx = self.size if self.full else self.ptr
        idx = np.random.randint(0, maxidx, size=batch)
        return (
            torch.tensor(self.obs[idx]),
            torch.tensor(self.act[idx]),
            torch.tensor(self.rew[idx]),
            torch.tensor(self.done[idx]),
            torch.tensor(self.next[idx]),
        )
