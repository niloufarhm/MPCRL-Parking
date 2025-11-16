from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		import numpy as np
		import torch

		ep_rewards, ep_successes, ep_lengths, ep_dists = [], [], [], []

		for i in range(self.cfg.eval_episodes):
			reset_out = self.env.reset()
			if isinstance(reset_out, tuple):
				obs, info = reset_out
			else:
				obs, info = reset_out, {}
			done, ep_reward, t = False, 0, 0

			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i == 0))

			ep_distances = []

			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t == 0, eval_mode=True)
				step_out = self.env.step(action)

				# Handle tuple returns (Gymnasium)
				if len(step_out) == 5:
					obs, reward, terminated, truncated, info = step_out
					done = terminated or truncated
				else:
					obs, reward, done, info = step_out

				ep_reward += reward
				t += 1

				# ✅ Log distance to goal if available
				if isinstance(info, dict):
					dist = None
					if "distance_to_goal" in info:
						dist = info["distance_to_goal"]
					elif "goal_distance" in info:
						dist = info["goal_distance"]
					elif "achieved_goal" in info and "desired_goal" in info:
						try:
							dist = float(np.linalg.norm(
								np.array(info["achieved_goal"]) - np.array(info["desired_goal"])
							))
						except Exception:
							pass
					if dist is not None:
						ep_distances.append(dist)

				if self.cfg.save_video:
					self.logger.video.record(self.env)

			ep_rewards.append(ep_reward)
			ep_successes.append(info.get('success', 0))
			ep_lengths.append(t)
			if ep_distances:
				ep_dists.append(np.mean(ep_distances))

			if self.cfg.save_video:
				self.logger.video.save(self._step)

		metrics = dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length=np.nanmean(ep_lengths),
		)
		if ep_dists:
			metrics["distance_to_goal"] = np.nanmean(ep_dists)

		return metrics

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		# 🧩 Handle Gymnasium (obs, info) tuple format
		if isinstance(obs, tuple) and len(obs) == 2:
			obs, info = obs
		else:
			info = {}

		# 🧩 Convert numpy or torch observation to TensorDict
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		elif isinstance(obs, np.ndarray):
			obs = torch.from_numpy(obs).float().unsqueeze(0)
		elif torch.is_tensor(obs):
			obs = obs.unsqueeze(0).cpu()
		else:
			raise TypeError(f"Unsupported obs type: {type(obs)}")

		# 🧩 Handle missing action/reward/terminated
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		elif isinstance(action, np.ndarray):
			action = torch.from_numpy(action).float()
		if reward is None:
			reward = torch.tensor(float('nan'))
		elif isinstance(reward, np.ndarray):
			reward = torch.tensor(reward.item(), dtype=torch.float32)
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		elif isinstance(terminated, bool):
			terminated = torch.tensor(float(terminated))

		# 🧩 Build TensorDict
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
			batch_size=(1,),
		)
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		print("start online trainer")
		train_metrics, done, eval_next = {}, True, False

		while self._step <= self.cfg.steps:
			# ✅ Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# ✅ If episode ended, reset env & log metrics
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info.get('terminated', False) and not self.cfg.episodic:
						raise ValueError(
							'Termination detected but you are not in episodic mode. '
							'Set `episodic=true` to enable support for terminations.'
						)

					# Log training episode metrics
					train_metrics.update(
						episode_reward=torch.tensor(
							[td['reward'] for td in self._tds[1:]]
						).sum(),
						episode_success=info.get('success', 0),
						episode_length=len(self._tds),
						episode_terminated=info.get('terminated', False)
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				# ✅ Reset environment & start new episode
				reset_out = self.env.reset()
				if isinstance(reset_out, tuple):
					obs, info = reset_out
				else:
					obs, info = reset_out, {}
				self._tds = [self.to_td(obs)]  # ✅ only once per episode
				done = False

			# ✅ Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds) == 1)
			else:
				action = self.env.rand_act()

			step_out = self.env.step(action)

			# ✅ Handle Gymnasium vs Gym returns
			if len(step_out) == 5:
				obs, reward, terminated, truncated, info = step_out
				done = terminated or truncated
			else:
				obs, reward, done, info = step_out

			# ✅ Append new transition (do NOT reset)
			self._tds.append(self.to_td(obs, action, reward, info.get('terminated', done)))

			# ✅ Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1

				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		self.logger.finish(self.agent)

