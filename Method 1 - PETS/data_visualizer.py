import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/parking_kinematics.npz")
obs = data['observations']
goals = data['goals']
actions = data['actions']

plt.figure(figsize=(8,8))
plt.scatter(obs[:,0], obs[:,1], alpha=0.4, label='Start positions')
plt.scatter(goals[:,0], goals[:,1], alpha=0.4, label='Goals')
plt.legend()
plt.title("Collected positions and goals")
plt.axis('equal')
plt.show()

plt.figure()
plt.hist(actions[:,0], bins=30, alpha=0.7, label='Steering')
plt.hist(actions[:,1], bins=30, alpha=0.7, label='Acceleration')
plt.legend()
plt.title("Action distribution")
plt.show()
