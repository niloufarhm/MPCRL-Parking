# TD-MPC2 Parking Agent 🚗🤖

A PyTorch implementation of the **TD-MPC2 (Temporal Difference Model Predictive Control v2)** reinforcement learning algorithm, trained to solve the **parking-v0** environment from `highway-env`.  
The agent combines **model-based planning (CEM)** with a **learned world model, value function,** and **actor policy** to learn efficient control.

---

## 📋 Features

- **TD-MPC2 algorithm** (latent world model + value learning)
- **CEM (Cross-Entropy Method)** model-predictive control
- **Replay buffer** for off-policy training
- **Hybrid actor/planner training** (starts with MPC, transitions to actor)
- **Checkpointing** + **JSONL logging** + **matplotlib visualizations**
- Compatible with **Highway-env**'s continuous car-parking task

---

## 🧩 Project Structure

```
.
├── agent.py           # Core TD-MPC2 model and planning logic
├── main.py            # Training script
├── replay_buffer.py   # Experience replay buffer
├── rollout.py         # Rollout and visualization script for trained agents
├── requirements.txt   # Python dependencies
├── checkpoints/       # Saved model checkpoints (created during training)
└── reports/           # Training logs and plots
```

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/tdmpc2-parking.git
cd tdmpc2-parking
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 🏋️‍♀️ Training

Run the main training loop:
```bash
python main.py
```

This will:
- Train the agent on `parking-v0`
- Save model checkpoints in `checkpoints/`
- Log results (losses, returns, evaluations) in `reports/training_log.jsonl`
- Generate summary plots in `reports/`

To stop and resume training later:
```bash
# Training resumes automatically from the latest checkpoint
python main.py
```

---

## 🎮 Rollout / Evaluation

Once trained, visualize the agent driving in real-time:
```bash
# Use the latest checkpoint and the learned actor
python rollout.py --latest --planner actor

# Or use the MPC planner
python rollout.py --latest --planner mpc
```

## 📊 Logs and Plots

All training data is logged to:
```
reports/training_log.jsonl
```

At the end of training (or upon interruption), the script automatically produces:
- `loss_over_steps.png`
- `episode_returns.png`
- `eval_returns.png`

---

## 🧠 How It Works

The TD-MPC2 agent consists of:

| Component | Description |
|------------|--------------|
| **WorldModel** | Encodes observations and predicts next latent state + reward |
| **ValueNet** | Estimates long-term value from latent states |
| **Actor** | Outputs actions directly from latent states (learned policy) |
| **CEM Planner** | Performs model-predictive control by sampling and optimizing action sequences |
| **Replay Buffer** | Stores experience tuples `(obs, act, reward, done, next_obs)` for learning |

The agent first trains using **CEM-planned actions**, then transitions to using its **actor policy** for faster inference.

---

## 💾 Checkpointing

Checkpoints are automatically saved as:
```
checkpoints/tdmpc2_step_XXXXX.pt
```

Each file contains:
- World model weights  
- Value and actor network weights  
- Optimizer state  
- Training step

---

## 📈 Example Results

After sufficient training (~200k steps), the agent typically learns to:
- Navigate efficiently to the parking goal
- Reduce collisions
- Stabilize at the goal position

Plots (saved in `reports/`):

- **Training loss**
- **Episode returns (moving average)**
- **Evaluation: MPC vs Actor performance**

---

## 🚀 References

- [TD-MPC2 Paper (Hansen et al., 2023)](https://arxiv.org/abs/2303.04939)
- [Highway-env Documentation](https://highway-env.readthedocs.io/en/latest/)
- [Gymnasium](https://gymnasium.farama.org/)
