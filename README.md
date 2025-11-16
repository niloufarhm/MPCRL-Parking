# Model-Based Reinforcement Learning for `parking-v0`
This repository contains the full implementation and experiments for our project:

**Model-Based Reinforcement Learning for Automated Parking  
A Comparative Study of PETS, TD-MPC2, and a Lightweight MPC–Actor Hybrid**

We evaluate **three model-based reinforcement learning (MBRL)** algorithms on the continuous-control **`parking-v0`** task from the `highway-env` package.  
Our goal is to understand how different world-models and planning strategies affect performance on a precise, goal-conditioned parking task.

---

# 🚗 Environment: `parking-v0`
We use the continuous-control environment from **highway-env**, a simplified but realistic simulated parking task.

### **Observation (6D)**  
- `(x, y)` — vehicle position  
- `θ` — heading  
- `v` — velocity  
- `(x_goal, y_goal)` — goal position  

### **Action (2D)**  
- steering  
- acceleration  

### **Reward**
Dense, proximity-based:
- distance to goal  
- heading alignment  
- low-speed at parking position  
- small penalties for high-speed or off-track actions


---

# 🧪 Methods Compared

## 1️⃣ PETS — Probabilistic Ensembles with Trajectory Sampling
- Ensemble of probabilistic neural networks  
- Uncertainty-aware predictions  
- Planning via **Cross-Entropy Method (CEM)**  
- HER + smart exploration significantly improve data quality  
- Produces smooth parking trajectories with heading alignment

**Result:**  
✔ Successfully solves parking  
✔ Smooth approach and braking  
✔ Most “realistic” behavior among the three methods  

---

## 2️⃣ TD-MPC2 — Latent World Model + Short-Horizon MPC
- Latent dynamics model  
- Actor + critic with TD backups  
- MPC planner with **horizon = 3**  
- Surprisingly strong generalization to the parking task  
- Stable training and small final goal distance

**Result:**  
✔ Fully solves the task  
✔ Robust even with default hyperparameters  
✖ Slightly jerky final motion; needs reward shaping for smoothness

---

## 3️⃣ Lightweight MPC–Actor Hybrid (ours)
- Simplified latent world-model  
- Short-horizon MPC for early supervision  
- Actor learns from planner  
- Very lightweight and fast

**Result:**  
❌ Does *not* solve the task  
Reasons:  
- insufficient model accuracy  
- too short planning horizon (6)  
- unstable gradients  
- inconsistent TD learning signals

---

# 📈 Training Results (Summary)

### PETS
- Converges after exploration improvements  
- Executes “C” or “S” shaped maneuvers to align heading  
- Zero velocity at goal for stable parking  

### TD-MPC2
- Consistency loss decreases monotonically  
- Actor entropy decreases steadily  
- Final goal error ≈ **0.12–0.14**

### Hybrid
- Loss oscillates heavily  
- No improvement in episode returns  
- Unable to generalize to multi-step maneuvers  

---

# ▶️ Running the Code

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Train PETS
```
python pets/train.py
```

### 3. Train TD-MPC2
```
python td_mpc2/train.py
```

### 4. Train Hybrid Method
```
python hybrid/train.py
```

### 5. Evaluate a model
```
python pets/eval.py --checkpoint checkpoints/pets_final.pt
```

---

# 📄 Full Report (LaTeX)
The full IEEE-style report is located in:

```
main.tex
```

---

# 📚 Citation
```
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}
```

---

# 🙌 Acknowledgements
- `highway-env` creators  
- Chua et al. for PETS  
- Hansen et al. for TD-MPC2 and their official implementation:
https://github.com/nicklashansen/tdmpc2
- Open-source RL community  
