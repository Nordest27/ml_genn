import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from snake_env import SnakeEnv
from performance_visualizer import PerformanceVisualizer
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


csv_file = open("ppo_training.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "episode",
    "reward",
    "reward_rate",
])

class ConvActorCritic(nn.Module):
    def __init__(self, in_channels, in_hw, hidden_dim, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, in_hw, in_hw)
            conv_out_dim = self._conv_forward(dummy).flatten(1).shape[1]

        self.fc = nn.Linear(conv_out_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._conv_forward(x).flatten(1)
        x = F.relu(self.fc(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def act(self, x):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, x, actions):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class VectorSnakeEnv:
    """N independent SnakeEnvs, auto-resetting on done -- a standard vec env."""

    def __init__(self, num_envs, **env_kwargs):
        self.num_envs = num_envs
        self.envs = [SnakeEnv(**env_kwargs) for _ in range(num_envs)]
        self.ep_return = np.zeros(num_envs, dtype=np.float32)
        self.ep_steps = np.zeros(num_envs, dtype=np.int64)
        self.obs = np.stack([e.reset() for e in self.envs], axis=0).astype(np.float32)

    def step(self, actions):
        finished = []  # list of (episode_return, episode_steps)
        next_obs = np.empty_like(self.obs)
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)

        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, d = env.step(int(a))
            self.ep_return[i] += r
            self.ep_steps[i] += 1
            rewards[i] = r
            if d:
                dones[i] = 1.0
                finished.append((float(self.ep_return[i]), int(self.ep_steps[i])))
                
                o = env.reset()
                self.ep_return[i] = 0.0
                self.ep_steps[i] = 0
            next_obs[i] = o.astype(np.float32)

        self.obs = next_obs
        return self.obs, rewards, dones, finished


def to_chw(obs_nhwc, device):
    # (N, H, W, C) -> (N, C, H, W)
    return torch.tensor(obs_nhwc, dtype=torch.float32, device=device).permute(0, 3, 1, 2)


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """
    Decoupled gamma/lambda, matching the thesis:
      - actor advantage : credit decays at lambda  (gamma kept only inside delta)
      - value target    : credit decays at gamma   (lambda plays no role in the critic)
    """
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)   # policy advantage (lambda)
    ret = torch.zeros_like(rewards)   # value-target accumulator (gamma)
    gae_pi = torch.zeros(N, device=rewards.device)
    gae_v  = torch.zeros(N, device=rewards.device)
    values_ext = torch.cat([values, next_value.unsqueeze(0)], dim=0)
    for t in reversed(range(T)):
        mask  = 1.0 - dones[t]
        # mask = 1.0
        delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
        gae_pi = delta + lam   * mask * gae_pi   # actor: lambda only  (decoupled)
        gae_v  = delta + gamma * mask * gae_v    # critic: gamma only  (lambda = 1)
        adv[t] = gae_pi
        ret[t] = gae_v
    returns = ret + values                       # gamma-consistent value target
    return adv, returns

@torch.no_grad()
def run_eval_episode(model, env_kwargs, device, max_steps=2000):
    """One sampled episode used only for visualizing value/prob traces -- not
    part of training, just a window into what the policy is doing."""
    env = SnakeEnv(**env_kwargs)
    obs = env.reset().astype(np.float32)
    done = False
    values, probs_hist = [], []
    total_reward, steps = 0.0, 0

    while not done and steps < max_steps:
        state = to_chw(obs[None], device)
        logits, value = model(state)
        probs = F.softmax(logits, dim=-1)
        action = torch.distributions.Categorical(probs).sample().item()
        values.append(value.item())
        probs_hist.append(probs.cpu().numpy().flatten())
        obs, reward, done = env.step(action)
        obs = obs.astype(np.float32)
        total_reward += reward
        steps += 1

    return total_reward, steps, values, probs_hist


def train_snake_agent_ppo(
    total_updates=20000,
    num_envs=16,
    rollout_steps=128,
    epochs=4,
    minibatch_size=512,
    gamma=0.99,
    lam=0.8,
    clip_eps=0.2,
    lr=3e-4,
    hidden_dim=512,
    entropy_coeff=0.01,
    value_coeff=0.5,
    grad_clip=0.5,
    eval_every=10,
):
    BOARD_SIZE = 5
    VISIBLE_RANGE = 5
    SCALE = 4
    INPUT_C = 3
    INPUT_HW = 20
    INPUT_SHAPE = (INPUT_HW, INPUT_HW, INPUT_C)
    N_ACTIONS = 4

    env_kwargs = dict(
        size=BOARD_SIZE,
        visible_range=VISIBLE_RANGE,
        scale=SCALE,
        wait_inc=0,
        inp_shape=INPUT_SHAPE,
    )

    vec_env = VectorSnakeEnv(num_envs, **env_kwargs)
    model = ConvActorCritic(INPUT_C, INPUT_HW, hidden_dim, N_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    viz = PerformanceVisualizer(window=100, stats_mode="mean+std+ema")

    best_eval_reward = -np.inf
    
    episode_counter = 0

    for update in range(total_updates):
        obs_buf = torch.zeros(rollout_steps, num_envs, INPUT_C, INPUT_HW, INPUT_HW, device=device)
        actions_buf = torch.zeros(rollout_steps, num_envs, dtype=torch.long, device=device)
        log_probs_buf = torch.zeros(rollout_steps, num_envs, device=device)
        values_buf = torch.zeros(rollout_steps, num_envs, device=device)
        rewards_buf = torch.zeros(rollout_steps, num_envs, device=device)
        dones_buf = torch.zeros(rollout_steps, num_envs, device=device)

        # ---- collect rollout (full backprop, no online-update restriction) ----
        for t in range(rollout_steps):
            state = to_chw(vec_env.obs, device)
            with torch.no_grad():
                action, log_prob, _, value = model.act(state)
            next_obs, reward, done, finished = vec_env.step(action.cpu().numpy())

            obs_buf[t] = state
            actions_buf[t] = action
            log_probs_buf[t] = log_prob
            values_buf[t] = value
            rewards_buf[t] = torch.tensor(reward, device=device)
            dones_buf[t] = torch.tensor(done, device=device)

            for ret, steps in finished:
                reward_rate = ret / max(steps, 1)

                csv_writer.writerow([
                    episode_counter,
                    ret,
                    reward_rate,
                ])

                episode_counter += 1
                viz.push_metrics(reward=ret)
                print(f"[update {update:05d}] episode return={ret:7.2f} steps={steps:4d}")

        with torch.no_grad():
            next_value = model.forward(to_chw(vec_env.obs, device))[1]

        advantages, returns = compute_gae(rewards_buf, values_buf, dones_buf, next_value, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # flatten (T, N) -> (T*N,)
        b_obs = obs_buf.reshape(-1, INPUT_C, INPUT_HW, INPUT_HW)
        b_actions = actions_buf.reshape(-1)
        b_log_probs = log_probs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        batch_size = b_obs.shape[0]
        idx = np.arange(batch_size)

        # ---- multiple epochs over the same batch -- the part eprop can't do ----
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = torch.as_tensor(idx[start:start + minibatch_size], device=device)

                new_log_probs, entropy, new_values = model.evaluate_actions(
                    b_obs[mb_idx], b_actions[mb_idx]
                )

                ratio = torch.exp(new_log_probs - b_log_probs[mb_idx])
                mb_adv = b_advantages[mb_idx]

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                mb_returns = b_returns[mb_idx]
                mb_old_values = b_values[mb_idx]
                value_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -clip_eps, clip_eps
                )
                value_loss = torch.max(
                    (new_values - mb_returns).pow(2),
                    (value_clipped - mb_returns).pow(2),
                ).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                

        if update % eval_every == 0:
            eval_reward, eval_steps, eval_values, eval_probs = run_eval_episode(model, env_kwargs, device)
            best_eval_reward = max(best_eval_reward, eval_reward)
            viz.push_metrics(values=eval_values, probs=eval_probs)

    csv_file.close()
    viz.close()


if __name__ == "__main__":
    train_snake_agent_ppo()