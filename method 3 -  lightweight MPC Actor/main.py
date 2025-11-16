import os, time, json, numpy as np, torch, gymnasium as gym, highway_env
from collections import deque
import matplotlib.pyplot as plt
from agent import TD_MPC2_Agent
from replay_buffer import ReplayBuffer

# ---------- Paths (edit to Drive paths if you mounted it) ----------
CKPT_DIR    = "checkpoints"
REPORTS_DIR = "reports"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORTS_DIR, "training_log.jsonl")  # json lines

# ---------- Training knobs ----------
TOTAL_STEPS       = 200_000
RANDOM_STEPS      = 2000        # pure random
USE_MPC_UNTIL     = 10_000       # after this, switch to actor for speed
UPDATES_START     = 1_000
UPDATES_PER_STEP  = 2
BATCH_SIZE        = 512
LOG_EVERY         = 1_000        # still print each 1k, but we log EVERY step/episode to file
SAVE_EVERY        = 5_000
EVAL_EVERY        = 10_000       # run an MPC eval episode
#PLOT_AT_END       = True         # make summary plots on finish
PLOT_AT_END = False

# ---------- Light CEM params (faster) ----------
CEM_HORIZON = 6
CEM_POP     = 64
CEM_ITERS   = 3
CEM_ELITE_FR  = 0.1
CEM_DISCOUNT  = 0.99

# ---------- Env ----------
def make_env():
    env = gym.make("parking-v0", render_mode="rgb_array")
    env.unwrapped.configure({
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x","y","vx","vy","cos_h","sin_h"],
            "scales": [100,100,5,5,1,1],
            "normalize": True,
        },
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15,
        "policy_frequency": 10,
        "offscreen_rendering": False,
    })
    env.reset()
    return env

def flatten_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)

# ---------- Actor action helper ----------
@torch.no_grad()
def policy_action(agent: TD_MPC2_Agent, obs, sample=True):
    device = agent.device
    z = agent.wm.encode(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
    a, dist = agent.actor(z)
    if not sample:
        a = torch.tanh(dist.mean)
    return a.squeeze(0).detach().cpu().numpy()

# ---------- Lightweight CEM ----------
@torch.no_grad()
def plan_cem_vectorized(agent: TD_MPC2_Agent, obs,
                        horizon=CEM_HORIZON, pop=CEM_POP,
                        iters=CEM_ITERS, elite_frac=CEM_ELITE_FR,
                        discount=CEM_DISCOUNT):
    device = agent.device
    act_dim = agent.actor.net[-1].out_features // 2
    z0 = agent.wm.encode(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
    z0 = z0.expand(pop, -1)  # repeat z0 for entire population

    elites = max(1, int(pop * elite_frac))
    mean = torch.zeros(horizon, act_dim, device=device)
    std  = torch.ones_like(mean) * 0.5

    for _ in range(iters):
        # Sample all actions in one go: (pop, horizon, act_dim)
        actions = torch.normal(mean.expand(pop, -1, -1), std.expand(pop, -1, -1))
        returns = torch.zeros(pop, device=device)
        gammas = torch.ones(pop, device=device)

        z = z0.clone()  # shape: (pop, latent_dim)
        for t in range(horizon):
            a_t = actions[:, t, :]  # (pop, act_dim)
            z, r = agent.wm.predict(z, a_t)
            returns += gammas * r.squeeze(-1)
            gammas *= discount

        returns += gammas * agent.val(z).squeeze(-1)
        top_idx = torch.topk(returns, elites).indices
        elite_actions = actions[top_idx]
        mean, std = elite_actions.mean(0), elite_actions.std(0) + 1e-4

    a0 = mean[0].clamp(-1, 1)
    return a0.detach().cpu().numpy()

# ---------- Checkpointing ----------
def save_checkpoint(step, agent: TD_MPC2_Agent):
    path = os.path.join(CKPT_DIR, f"tdmpc2_step_{step}.pt")
    torch.save({
        "step": step,
        "wm": agent.wm.state_dict(),
        "val": agent.val.state_dict(),
        "actor": agent.actor.state_dict(),
        "opt": agent.opt.state_dict(),
    }, path)
    print(f"[checkpoint] saved → {path}")

def load_latest_checkpoint(agent: TD_MPC2_Agent):
    ckpts = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")]
    if not ckpts:
        print("[checkpoint] none found, starting fresh")
        return 1
    ckpts.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))  # tdmpc2_step_XXXXX.pt
    latest = os.path.join(CKPT_DIR, ckpts[-1])
    print(f"[resume] loading {latest}")
    ckpt = torch.load(latest, map_location="cpu")
    agent.wm.load_state_dict(ckpt["wm"])
    agent.val.load_state_dict(ckpt["val"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.opt.load_state_dict(ckpt["opt"])
    start = int(ckpt["step"]) + 1
    print(f"[resume] resumed from step {start-1}")
    return start

# ---------- JSON logging ----------
def log_jsonl(obj):
    obj["ts"] = time.time()
    with open(REPORT_PATH, "a") as f:
        f.write(json.dumps(obj) + "\n")

def log_step(step, losses, avg_return):
    log_jsonl({
        "type": "step",
        "step": int(step),
        "model_loss": float(losses["model_loss"]),
        "value_loss": float(losses["v_loss"]),
        "actor_loss": float(losses["actor_loss"]),
        "total_loss": float(losses["total_loss"]),
        "avg_return": float(avg_return)
    })

def log_episode(ep_idx, step, ep_return, ep_len, avg_return):
    log_jsonl({
        "type": "episode",
        "episode": int(ep_idx),
        "step_end": int(step),
        "return": float(ep_return),
        "length": int(ep_len),
        "avg_return": float(avg_return)
    })

def log_eval(step, ret_mpc, len_mpc, ret_actor, len_actor):
    log_jsonl({
        "type": "eval",
        "step": int(step),
        "mpc_return": float(ret_mpc),
        "mpc_length": int(len_mpc),
        "actor_return": float(ret_actor),
        "actor_length": int(len_actor)
    })

# ---------- Eval ----------
def run_eval_episode(env, agent, use_mpc=True, max_steps=400):
    ob, _ = env.reset()
    ob = flatten_obs(ob)
    ret, t, done = 0.0, 0, False
    while not done and t < max_steps:
        a = plan_cem_vectorized(agent, ob) if use_mpc else policy_action(agent, ob, sample=False)
        ob, r, term, trunc, _ = env.step(a)
        ret += r
        ob = flatten_obs(ob)
        done = term or trunc
        t += 1
    return ret, t

# ---------- Plotting ----------
def make_plots():
    steps, losses, avg_steps = [], [], []
    ep_idx, ep_returns, ep_avg, ep_steps = [], [], [], []
    eval_steps, mpc_ret, actor_ret = [], [], []

    # parse jsonl
    with open(REPORT_PATH, "r") as f:
        for line in f:
            rec = json.loads(line)
            t = rec.get("type", "")
            if t == "step":
                steps.append(rec["step"])
                losses.append(rec["loss"] if rec["loss"] is not None else None)
                avg_steps.append(rec["avg_return"])
            elif t == "episode":
                ep_idx.append(rec["episode"])
                ep_returns.append(rec["return"])
                ep_avg.append(rec["avg_return"])
                ep_steps.append(rec["step_end"])
            elif t == "eval":
                eval_steps.append(rec["step"])
                mpc_ret.append(rec["mpc_return"])
                actor_ret.append(rec["actor_return"])

    def _savefig(name):
        path = os.path.join(REPORTS_DIR, name)
        plt.savefig(path, bbox_inches="tight")
        print(f"[plot] saved → {path}")
        plt.clf()

    # 1) Loss over steps
    plt.figure(figsize=(8,4))
    ys = [v for v in losses if v is not None]
    xs = [s for s,v in zip(steps, losses) if v is not None]
    if xs:
        plt.plot(xs, ys)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    _savefig("loss_over_steps.png")

    # 2) Episode returns (scatter) + rolling avg (line)
    plt.figure(figsize=(8,4))
    if ep_idx:
        plt.scatter(ep_steps, ep_returns, s=8, alpha=0.5, label="episode return")
    if ep_steps:
        # 100-episode moving average of returns:
        import math
        win = 100
        if len(ep_returns) >= 2:
            mov = []
            for i in range(len(ep_returns)):
                lo = max(0, i - win + 1)
                mov.append(np.mean(ep_returns[lo:i+1]))
            plt.plot(ep_steps, mov, linewidth=2, label=f"{win}-episode moving avg")
    plt.title("Episode Returns")
    plt.xlabel("Env Step (end of episode)")
    plt.ylabel("Return")
    plt.legend()
    _savefig("episode_returns.png")

    # 3) Actor vs MPC eval returns
    plt.figure(figsize=(8,4))
    if eval_steps:
        plt.plot(eval_steps, mpc_ret, label="MPC eval return")
        plt.plot(eval_steps, actor_ret, label="Actor eval return")
        plt.legend()
    plt.title("Eval: MPC vs Actor")
    plt.xlabel("Step")
    plt.ylabel("Return")
    _savefig("eval_returns.png")

# ---------- Main ----------
def main():
    env = make_env()
    obs_dim, act_dim = 12, 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    agent = TD_MPC2_Agent(obs_dim, act_dim, device=device)
    buffer = ReplayBuffer(obs_dim, act_dim)
    start_step = load_latest_checkpoint(agent)

    o, _ = env.reset()
    o = flatten_obs(o)
    last_loss = None
    episode_return, episode_len = 0.0, 0
    recent_returns = deque(maxlen=10)
    episodes_seen = 0

    print(f"[info] Training started. random={RANDOM_STEPS}, mpc_until={USE_MPC_UNTIL}, save_every={SAVE_EVERY}")

    try:
        for step in range(start_step, TOTAL_STEPS + 1):
            # --- action selection (hybrid) ---
            if step < RANDOM_STEPS:
                a = np.random.uniform(-1, 1, act_dim)
            elif step < USE_MPC_UNTIL:
                a = plan_cem_vectorized(agent, o)  # MPC (light)
            else:
                a = policy_action(agent, o, sample=True)  # fast actor

            a = np.clip(a, -1.0, 1.0)

            # --- env step ---
            o2, r, term, trunc, _ = env.step(a)
            done = term or trunc or (episode_len >= 100)
            episode_return += r
            episode_len += 1

            o2 = flatten_obs(o2)
            buffer.add(o, a, r, done, o2)
            o = o2 if not done else flatten_obs(env.reset()[0])

            # --- learn ---
            if step > UPDATES_START:
                for _ in range(UPDATES_PER_STEP):
                    batch = buffer.sample(BATCH_SIZE)
                    res = agent.update(batch)

                # compute rolling return average
                avg_ret = float(np.mean(recent_returns)) if recent_returns else 0.0

                # log losses and average return
                #log_step(step, res, avg_ret)
                if step % 20 == 0: #LOG_EVERY == 0:
                    log_step(step, res, avg_ret)

            # --- episode end ---
            if done:
                episodes_seen += 1
                recent_returns.append(episode_return)
                # Keep only the last 10 returns for rolling average
                #recent_returns = recent_returns[-10:]
                avg_ret = float(np.mean(recent_returns))

                print(f"[episode {episodes_seen}] return={episode_return:.2f} len={episode_len} avg(10)={avg_ret:.2f}")
                log_episode(episodes_seen, step, episode_return, episode_len, avg_ret)

                # Reset for next episode
                episode_return = 0.0
                episode_len = 0
                o, _ = env.reset()
                o = flatten_obs(o)

            # --- console heartbeat ---
            if step % LOG_EVERY == 0:
                loss_str = f"{last_loss:.4f}" if last_loss is not None else "N/A"
                print(f"[step {step}] loss={loss_str}, avg_return={avg_ret:.2f}")

            # --- checkpoint ---
            if step % SAVE_EVERY == 0:
                save_checkpoint(step, agent)

            # --- periodic eval ---
            if step % EVAL_EVERY == 0:
                ret_mpc, len_mpc     = run_eval_episode(env, agent, use_mpc=True)
                ret_actor, len_actor = run_eval_episode(env, agent, use_mpc=False)
                print(f"[eval @ {step}] MPC  return={ret_mpc:.2f} len={len_mpc} | "
                      f"Actor return={ret_actor:.2f} len={len_actor}")
                log_eval(step, ret_mpc, len_mpc, ret_actor, len_actor)

        print(f"[info] Training complete. Total steps: {TOTAL_STEPS}")

    except KeyboardInterrupt:
        print("\n[info] Interrupted — saving checkpoint...")
        save_checkpoint(step, agent)

    finally:
        env.close()
        print(f"[info] Environment closed. Logs at {REPORT_PATH}")
        if PLOT_AT_END:
            print("[plot] generating plots...")
            make_plots()

if __name__ == "__main__":
    main()
