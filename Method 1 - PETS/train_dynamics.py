import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
from config import buffer_file, batch_size, epochs, lr, state_dim, action_dim, device, models_dir
from utils import load_buffer
from models import ResidualDynamics
from tqdm import tqdm
if __name__ == '__main__':
    print(f"Using device: {device}")
    print('Loading buffer:', buffer_file)
    obs, goals, acts, next_observations, success_flags = load_buffer(buffer_file)

    print(f"--- Loaded {len(obs)} total samples from buffer. ---")

    print("Converting data to Tensors and moving to device...")
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    goals_tensor = torch.tensor(goals, dtype=torch.float32).to(device)
    acts_tensor = torch.tensor(acts, dtype=torch.float32).to(device)
    next_obs_tensor = torch.tensor(next_observations, dtype=torch.float32).to(device)
    success_flags_tensor = torch.tensor(success_flags, dtype=torch.bool).to(device)

    dataset = TensorDataset(obs_tensor, goals_tensor, acts_tensor, next_obs_tensor, success_flags_tensor)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=0,
                        pin_memory=False)

    print("Data loading complete. Starting training...")

    fwd = ResidualDynamics().to(device)
    opt = torch.optim.Adam(fwd.parameters(), lr=lr, weight_decay=1e-5)

    for ep in range(epochs):
        total_loss = 0.0
        fwd.train()
        pbar = tqdm(loader, desc=f"Epoch {ep + 1}/{epochs}")

        for s, g, a, s_next, success_flags in pbar:
            s_pred = fwd(s, a)

            sample_loss = (s_pred - s_next).pow(2).mean(dim=1)
            weights = torch.where(success_flags > 0.5, 3.0, 1.0)
            loss = (weights * sample_loss).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(fwd.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {ep + 1}/{epochs} Complete, Average Loss: {avg_loss:.6f}")

    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, 'fwd_state.pth')
    torch.save(fwd.state_dict(), path)
    print('Saved model to', path)