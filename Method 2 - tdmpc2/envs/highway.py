import highway_env
import gymnasium as gym
import numpy as np
from envs.wrappers.tensor import TensorWrapper

import torch

class FlattenObservationWrapper(gym.Wrapper):
    """Flattens dict observations and adds distance_to_goal metric."""
    def __init__(self, env):
        super().__init__(env)
        obs, info = env.reset()
        flat_obs = self._flatten(obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32
        )
        self.max_episode_steps = getattr(env, "_max_episode_steps", 1000)

    def _flatten(self, obs):
        if isinstance(obs, dict):
            obs = np.concatenate([
                obs.get("observation", np.array([], dtype=np.float32)).ravel(),
                obs.get("achieved_goal", np.array([], dtype=np.float32)).ravel(),
                obs.get("desired_goal", np.array([], dtype=np.float32)).ravel(),
            ])
        return obs.astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        flat_obs = self._flatten(obs)
        if "achieved_goal" in obs and "desired_goal" in obs:
            info["distance_to_goal"] = float(np.linalg.norm(
                obs["achieved_goal"][:2] - obs["desired_goal"][:2]
            ))
        # ✅ Return as torch tensor
        return torch.from_numpy(flat_obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        flat_obs = self._flatten(obs)
        if "achieved_goal" in obs and "desired_goal" in obs:
            info["distance_to_goal"] = float(np.linalg.norm(
                obs["achieved_goal"][:2] - obs["desired_goal"][:2]
            ))
        done = terminated or truncated
        # ✅ Return obs as torch tensor
        return torch.from_numpy(flat_obs), float(reward), done, info

def make_env(cfg):
    print("[INFO] Loading HighwayEnv: parking-v0")
    env = gym.make("parking-v0", render_mode="rgb_array")
    print("[INFO] Highway Parking-v0 environment ready.")

    env = FlattenObservationWrapper(env)
    env = TensorWrapper(env)
    return env
