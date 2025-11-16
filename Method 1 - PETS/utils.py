import numpy as np
import torch
import os

def save_buffer(path, observations, goals, actions, next_observations,success_flags):
    np.savez_compressed(path,
                        observations=np.array(observations, dtype=np.float32),
                        goals=np.array(goals, dtype=np.float32),
                        actions=np.array(actions, dtype=np.float32),
                        next_observations=np.array(next_observations, dtype=np.float32),
                        success_flags=np.array(success_flags, dtype=bool))

def load_buffer(path):
    d = np.load(path)
    return d['observations'], d['goals'], d['actions'], d['next_observations'],d['success_flags']
