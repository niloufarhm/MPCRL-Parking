#
# collect_data.py
#
import numpy as np
from config import num_episodes, max_episode_steps, buffer_file
from env_utils import make_env, obs_to_state
from utils import save_buffer

import multiprocessing as mp
import os
import time
from tqdm import tqdm
def collect_smart_data_with_HER(env, num_episodes, max_steps, her_ratio=0.8,
                                success_boost=5, action_repeat=4, worker_seed=None):
    all_obs, all_goals, all_actions, all_next_obs, all_success_flags = [], [], [], [], []
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)

    for ep in range(num_episodes):
        ep_seed = worker_seed + ep if worker_seed is not None else None
        obs, info = env.reset(seed=ep_seed)

        episode_transitions = []
        success = False

        for step in range(max_steps):
            if step % action_repeat == 0:
                steering = np.random.uniform(-1, 1)
                rand_val = np.random.rand()
                if rand_val < 0.2:
                    accel = -1.0
                elif rand_val < 0.4:
                    accel = 1.0
                elif rand_val < 0.5:
                    accel = 0.0
                else:
                    accel = np.random.uniform(-1, 1)
                a = np.array([steering, accel], dtype=np.float32)

            obs_next, _, terminated, truncated, info = env.step(a)
            transition = {
                "obs": obs_to_state(obs["observation"]),
                "goal": obs["desired_goal"],
                "action": a,
                "next_obs": obs_to_state(obs_next["observation"]),
                "achieved_goal": obs_next["achieved_goal"],
            }
            episode_transitions.append(transition)
            obs = obs_next
            if terminated or truncated:
                success = info.get("is_success", False)
                break

        repeat_factor = success_boost if success else 1
        n_trans = len(episode_transitions)

        for i, trans in enumerate(episode_transitions):
            # 1. Original goal
            for _ in range(repeat_factor):
                all_obs.append(trans["obs"])
                all_goals.append(trans["goal"])
                all_actions.append(trans["action"])
                all_next_obs.append(trans["next_obs"])
                all_success_flags.append(int(success))

            # 2. HER goals
            n_her = int(her_ratio * 2)
            future_idxs = np.random.choice(range(i, n_trans), size=min(n_her, n_trans - i), replace=False)
            for idx in future_idxs:
                her_goal = episode_transitions[idx]["achieved_goal"]
                for _ in range(repeat_factor):
                    all_obs.append(trans["obs"])
                    all_goals.append(her_goal)
                    all_actions.append(trans["action"])
                    all_next_obs.append(trans["next_obs"])
                    all_success_flags.append(int(success))

    return all_obs, all_goals, all_actions, all_next_obs, all_success_flags


def collect_worker(worker_id, num_episodes_per_worker, worker_seed):
    env = make_env(render=False)

    data = collect_smart_data_with_HER(
        env, num_episodes_per_worker, max_episode_steps,
        her_ratio=0.8, success_boost=5, action_repeat=4,
        worker_seed=worker_seed
    )

    env.close()
    return data


def collect_worker_star(args):
    return collect_worker(*args)


if __name__ == "__main__":
    from config import seed
    start_time = time.time()

    num_workers = max(1, os.cpu_count() - 1)
    total_episodes = num_episodes
    episodes_per_worker = total_episodes // num_workers
    remainder_episodes = total_episodes % num_workers
    current_seed = seed
    worker_args = []
    for i in range(num_workers):
        eps_to_run = episodes_per_worker
        if i < remainder_episodes:
            eps_to_run += 1

        worker_args.append((i, eps_to_run, current_seed))
        current_seed += eps_to_run
    print(f"Starting {num_workers} workers to collect {total_episodes} total episodes...")

    results = []
    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm(pool.imap_unordered(collect_worker_star, worker_args), total=len(worker_args),
                    desc="Collecting Data")
        for result in pbar:
            results.append(result)

    print("All workers finished. Concatenating results...")

    # --- Concatenate Results ---
    final_obs, final_goals, final_acts, final_obs2, final_success = [], [], [], [], []

    for (obs, goals, acts, obs2, success_flags) in results:
        final_obs.extend(obs)
        final_goals.extend(goals)
        final_acts.extend(acts)
        final_obs2.extend(obs2)
        final_success.extend(success_flags)

    print(f"Saving {len(final_obs)} total samples to {buffer_file} ...")
    save_buffer(buffer_file, final_obs, final_goals, final_acts, final_obs2, final_success)

    end_time = time.time()
    print(f"Done. Total time: {end_time - start_time:.2f} seconds")