import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Import all required components ---
from config import device, planning_horizon, cem_iters, cem_pop, cem_elite_frac, models_dir, buffer_file, seed
from env_utils import make_env, obs_to_state
from models import ResidualDynamics  # Assuming ResidualDynamics
from utils import load_buffer, save_buffer
from planning import CEMPlannerState, success_state


def run_episode_with_parking_debug(env, planner, episode_seed):
    # 1. Get start state (s) and goal (sg) directly from the environment
    obs, _ = env.reset(seed=episode_seed)
    s = obs_to_state(obs['observation'])
    sg = obs_to_state(obs['desired_goal'])  # Goal is constant for the episode

    # Lists for plotting and logging
    trajectory_coords = []
    actions_taken = []
    distances = []
    speeds = []
    rewards = []

    episode_data_buffer = []

    planner.prev_solution = None
    success = False

    print(f"--- Running Episode ---")
    print(f"Start: pos=[{s[0]:.3f}, {s[1]:.3f}]")
    print(f"Goal:  pos=[{sg[0]:.3f}, {sg[1]:.3f}]")

    for t in range(150):  # Max 150 steps
        # Store data for plotting
        trajectory_coords.append(s[:2].copy())
        distance = np.linalg.norm(s[:2] - sg[:2])
        distances.append(distance)
        speed = np.linalg.norm(s[2:4])
        speeds.append(speed)

        # 1. Check for custom success (strict definition)
        if success_state(s, sg):
            print(f'🎉 CUSTOM SUCCESS at step {t}')
            success = True
            break

        # 2. Plan action (the MPC core)
        a = planner.plan(s, sg, horizon=planning_horizon, iters=cem_iters,
                         pop=cem_pop, elite_frac=cem_elite_frac, device=device)

        a = np.clip(a, env.action_space.low, env.action_space.high)
        actions_taken.append(a.copy())

        # 3. Execute action in environment
        # 'terminated', 'truncated', and 's_next' are defined here
        obs, reward, terminated, truncated, info = env.step(a)
        s_next = obs_to_state(obs['observation'])
        rewards.append(reward)

        # 4. Store (s, a, s_next) data for refinement
        episode_data_buffer.append({
            "obs": s.copy(),
            "goal": sg.copy(),  # Use the episode's constant goal
            "action": a.copy(),
            "next_obs": s_next.copy()
        })

        s = s_next  # Update state for the next loop

        # 5. Check for environment termination
        if terminated or truncated:
            print(f'Episode terminated by env at step {t}. Info: {info}')
            break

    # --- Plotting Section ---
    trajectory = np.array(trajectory_coords)
    actions_taken = np.array(actions_taken)
    distances = np.array(distances)
    speeds = np.array(speeds)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Trajectory
    if len(trajectory) > 0:
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=150, marker='s', label='Start', zorder=5)
    ax1.scatter(sg[0], sg[1], color='red', s=200, marker='*', label='Goal Position', zorder=5)

    # 'arrow_len' is defined here
    arrow_len = 0.25  # Length of the arrow for display

    # --- Corrected sin/cos bug ---
    # sg[4] = cos(h), sg[5] = sin(h) (assuming standard 'KinematicsGoal' obs)
    goal_dx = arrow_len * sg[4]  # U component (Cosine)
    goal_dy = arrow_len * sg[5]  # V component (Sine)
    # -----------------------------

    ax1.quiver(sg[0], sg[1], goal_dx, goal_dy, color='red', scale=1, scale_units='xy', angles='xy', zorder=6,
               width=0.007, label='Goal Heading')

    circles = [0.2, 0.5, 1.0]  # Standard radii
    for r in circles:
        ax1.add_patch(plt.Circle((sg[0], sg[1]), r, fill=False, alpha=0.3, linestyle='--', label=f'{r}m radius'))
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Parking Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Distance
    ax2.plot(distances, 'g-', linewidth=2)
    ax2.set_xlabel('Step');
    ax2.set_ylabel('Distance to Goal (m)');
    ax2.set_title('Distance Progress')
    ax2.grid(True, alpha=0.3);
    ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Close threshold');
    ax2.legend()

    # 3. Actions
    if len(actions_taken) > 0:
        ax3.plot(actions_taken[:, 0], 'r-', label='Steering', linewidth=2)
        ax3.plot(actions_taken[:, 1], 'b-', label='Acceleration', linewidth=2)
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5);
        ax3.axhline(y=-1.0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Step');
        ax3.set_ylabel('Action Value');
        ax3.set_title('Actions Over Time');
        ax3.legend();
        ax3.grid(True, alpha=0.3)

    # 4. Velocity
    if len(speeds) > 0:
        ax4.plot(speeds, 'm-', linewidth=2, label='Speed (m/s)')
        ax4.set_xlabel('Step');
        ax4.set_ylabel('Speed (m/s)');
        ax4.set_title('Velocity Profile');
        ax4.grid(True, alpha=0.3);
        ax4.legend()

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/parking_run_{int(time.time())}.png', dpi=150, bbox_inches='tight')
    plt.show()

    total_reward = np.sum(rewards)
    print(f"\n--- Episode Finished ---")
    print(f"Total Environment Reward Collected: {total_reward:.2f}")

    return success, episode_data_buffer


def append_failed_data_to_buffer(episode_data, buffer_path):
    """
    Appends data from a failed episode to the main buffer.
    """
    print(f"Appending {len(episode_data)} failed transitions to buffer...")
    new_obs, new_goals, new_actions, new_next_obs = [], [], [], []
    new_success_flags = [0] * len(episode_data)  # This was a failed episode

    for transition in episode_data:
        new_obs.append(transition['obs'])
        new_goals.append(transition['goal'])
        new_actions.append(transition['action'])
        new_next_obs.append(transition['next_obs'])

    try:
        old_obs, old_goals, old_acts, old_next_obs, old_success = load_buffer(buffer_path)
        if not isinstance(old_obs, np.ndarray) or old_obs.size == 0:
            print("Buffer was empty or invalid. Starting fresh.")
            raise FileNotFoundError  # Jump to except block
        old_data = [old_obs, old_goals, old_acts, old_next_obs, old_success]

    except Exception as e:
        print(f"Buffer file '{buffer_path}' not found or corrupted. Creating new buffer. Error: {e}")
        # Get dims from the first new data point
        s_dim = new_obs[0].shape[0]
        g_dim = new_goals[0].shape[0]
        a_dim = new_actions[0].shape[0]
        old_data = [np.empty((0, s_dim), dtype=np.float32),
                    np.empty((0, g_dim), dtype=np.float32),
                    np.empty((0, a_dim), dtype=np.float32),
                    np.empty((0, s_dim), dtype=np.float32),
                    np.empty((0,), dtype=bool)]  # Use bool

    [old_obs, old_goals, old_acts, old_next_obs, old_success] = old_data

    # 3. Concatenate old and new data
    all_obs = np.concatenate([old_obs, np.array(new_obs, dtype=np.float32)])
    all_goals = np.concatenate([old_goals, np.array(new_goals, dtype=np.float32)])
    all_actions = np.concatenate([old_acts, np.array(new_actions, dtype=np.float32)])
    all_next_obs = np.concatenate([old_next_obs, np.array(new_next_obs, dtype=np.float32)])
    all_success = np.concatenate([old_success, np.array(new_success_flags, dtype=bool)])  # Use bool

    # 4. Save the combined buffer
    save_buffer(buffer_path, all_obs, all_goals, all_actions, all_next_obs, all_success)
    print(f"Successfully added failed data. New buffer size: {len(all_obs)}")


if __name__ == '__main__':
    print('Setting up parking environment...')
    env = make_env(render=True)

    print('Loading dynamics model...')
    fwd = ResidualDynamics().to(device)
    model_path = os.path.join(models_dir, 'fwd_state.pth')
    fwd.load_state_dict(torch.load(model_path, map_location=device))
    fwd.eval()  # Set model to evaluation mode

    act_low, act_high = env.action_space.low, env.action_space.high
    planner = CEMPlannerState(fwd, act_low, act_high, action_dim=env.action_space.shape[0])

    # No more manual goal creation.

    print("\n=== RUNNING PARKING EPISODE ===")
    # The goal (sg) will be read from env.reset() inside the function
    # Use the global 'seed' from config for the first run, then random
    episode_seed = seed
    success, episode_data = run_episode_with_parking_debug(env, planner, episode_seed)

    print(f'\nResult: {"SUCCESS" if success else "FAILED"}')

    # Iterative Refinement: If it failed, add the data to the buffer
    if not success and len(episode_data) > 0:
        try:
            append_failed_data_to_buffer(episode_data, buffer_file)
            print("\n---")
            print("Next step: Re-run 'train_dynamics.py' to improve your model!")
            print("---")
        except Exception as e:
            print(f"Could not append data to buffer: {e}")

    env.close()