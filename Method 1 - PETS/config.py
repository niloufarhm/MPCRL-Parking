import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = 'data'
models_dir = 'models'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

buffer_file = os.path.join(data_dir, 'parking_kinematics.npz')

# data collection
num_episodes = 2000
max_episode_steps = 100

# training
state_dim = 6
action_dim = 2
batch_size = 256
epochs = 100
lr = 5e-4
seed = 3
# planning
planning_horizon = 60
cem_iters = 10
cem_pop = 4096
cem_elite_frac = 0.1


