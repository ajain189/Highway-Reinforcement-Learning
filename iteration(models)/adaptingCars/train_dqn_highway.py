import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.next_states[self.idx] = next_state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        states = torch.tensor(self.states[idxs], dtype=torch.float32, device=device)
        next_states = torch.tensor(self.next_states[idxs], dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions[idxs], dtype=torch.int64, device=device)
        rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones[idxs], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones


class AdvancedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.vehicles_ahead = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.vehicles_ahead = set()
        # Initialize vehicles ahead
        if hasattr(self.env.unwrapped, 'vehicle') and hasattr(self.env.unwrapped, 'road'):
            ego_x = self.env.unwrapped.vehicle.position[0]
            for v in self.env.unwrapped.road.vehicles:
                if v is not self.env.unwrapped.vehicle and v.position[0] > ego_x:
                    self.vehicles_ahead.add(v)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract Ego State (Raw values, Relative Obs)
        # obs shape: (vehicles_count, features)
        # features: [presence, x, y, vx, vy, cos_h, "sin_h"]
        # With absolute=False, ego x is always 0 (relative to itself)
        ego = obs[0]
        # ego_x = float(ego[1]) # Should be ~0
        # ego_y = float(ego[2])
        ego_vx = float(ego[3])
        ego_vy = float(ego[4])
        speed = float(np.sqrt(ego_vx**2 + ego_vy**2))
        
        # Find distance to front vehicle
        min_front_gap = 100.0 # Default max
        
        for i in range(1, len(obs)):
            if obs[i][0] == 0: # Not present
                continue
            
            # Relative coordinates
            rel_x = float(obs[i][1])
            rel_y = float(obs[i][2])
            
            # Check if in same lane (approx width 4m) and ahead
            # Since relative, ahead means rel_x > 0
            if rel_x > 0 and abs(rel_y) < 2.0:
                if rel_x < min_front_gap:
                    min_front_gap = rel_x

        # --- OVERTAKING LOGIC ---
        overtaking_bonus = 0.0
        if hasattr(self.env.unwrapped, 'vehicle') and hasattr(self.env.unwrapped, 'road'):
            # Use raw position from env for accuracy (absolute position still exists in env)
            raw_ego_x = self.env.unwrapped.vehicle.position[0]
            
            # Optimize: Convert road vehicles to set for O(1) lookups
            current_vehicles = set(self.env.unwrapped.road.vehicles)
            
            passed_vehicles = set()
            # Check if we passed any vehicle that was ahead
            for v in list(self.vehicles_ahead):
                if v.position[0] < raw_ego_x:
                    # We passed it!
                    overtaking_bonus += 1.0
                    passed_vehicles.add(v)
                # Stop tracking if despawned
                if v not in current_vehicles:
                    passed_vehicles.add(v)
            
            self.vehicles_ahead -= passed_vehicles
            
            # Add new vehicles ahead
            for v in current_vehicles:
                if v is not self.env.unwrapped.vehicle and v.position[0] > raw_ego_x:
                    self.vehicles_ahead.add(v)

        # --- REWARD SHAPING ---
        # Start with the base environment reward (Speed + Collision)
        shaped_reward = reward

        # 1. SURVIVAL BONUS (Steady)
        if not terminated:
            shaped_reward += 0.5

        # 2. HEADWAY SHAPING (Relative)
        if min_front_gap > 40.0:
            # Clear road ahead
            shaped_reward += 0.2
        elif min_front_gap < 15.0:
            # Dangerous tailgating (Speed is fixed, so just gap matters)
            shaped_reward -= 0.5

        # 3. LANE CHANGE COST (Free to weave)
        # Actions: 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT
        # No longitudinal actions available
        if action == 0 or action == 2:
            shaped_reward -= 0.0

        # 4. OVERTAKING (Aggressive)
        shaped_reward += overtaking_bonus * 2.0

        return obs, shaped_reward, terminated, truncated, info


def normalize_obs(obs):
    # obs shape: (vehicles_count, features)
    # features: [presence, x, y, vx, vy, cos_h, sin_h]
    obs = np.array(obs, dtype=np.float32)
    # presence is 0 or 1, no need to normalize
    obs[..., 1] /= 100.0 # x
    obs[..., 2] /= 10.0  # y
    obs[..., 3] /= 40.0  # vx
    obs[..., 4] /= 40.0  # vy
    # cos_h, sin_h are -1 to 1, no need to normalize
    return obs

def flatten_obs(obs):
    return normalize_obs(obs).reshape(-1)

def make_env(render_mode=None):
    config = {
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": False, # FIXED SPEED: Agent cannot slow down
            "target_speeds": [30, 35] # Range required to avoid div by zero error
        },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False, # RELATIVE OBS: See cars relative to ego
            "order": "sorted",
            "normalize": False # Use raw physics values
        },
        "policy_frequency": 15,
        "duration": 60,
        "vehicles_count": 30,
        "controlled_vehicles": 1,
        "lanes_count": 4,
        # Balanced Base Rewards
        "right_lane_reward": 0.0, # Don't care about lane
        "lane_change_reward": 0.0, # Free weaving
        "collision_reward": -10.0, # Decent penalty
        "high_speed_reward": 0.4,
        "reward_speed_range": [20, 35],
        "speed_limit": 35,
        "normalize_reward": True
    }
    env = gym.make("highway-v0", config=config, render_mode=render_mode)
    env = AdvancedRewardWrapper(env)
    return env

def train_dqn():
    env = make_env(render_mode=None)
    obs, info = env.reset()
    state = flatten_obs(obs)
    state_dim = state.shape[0]
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(state_dim, num_actions).to(device)
    target_net = DQN(state_dim, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
    buffer = ReplayBuffer(capacity=120000, state_dim=state_dim)
    writer = SummaryWriter(log_dir="runs/dqn_highway")

    gamma = 0.99
    batch_size = 64
    start_learning = 2000 # Reverted
    target_update_freq = 2000
    max_episodes = 2000 # Increased from 800
    max_steps = 10000

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 200000 # Increased to match longer training

    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    episode_mean_losses = []

    for episode in range(max_episodes):
        obs, info = env.reset()
        state = flatten_obs(obs)
        done = False
        ep_reward = 0.0
        steps = 0
        running_losses = []

        while not done and steps < max_steps:
            epsilon = max(
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end) * total_steps / epsilon_decay_steps
            )

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(s)
                    action = int(torch.argmax(q_values, dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = flatten_obs(next_obs)
            done_flag = terminated or truncated

            clipped_reward = max(min(reward, 5.0), -5.0)
            buffer.add(state, action, clipped_reward, next_state, done_flag)

            state = next_state
            ep_reward += reward
            steps += 1
            total_steps += 1

            if buffer.size >= start_learning and total_steps % 4 == 0:
                states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size, device)

                q_values = policy_net(states_b)
                q_values = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_actions = policy_net(next_states_b).argmax(1)
                    next_q_values = target_net(next_states_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    targets = rewards_b + gamma * (1.0 - dones_b) * next_q_values

                loss = nn.MSELoss()(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                running_losses.append(loss.item())

            if total_steps % target_update_freq == 0 and total_steps > 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done_flag:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        mean_loss = float(np.mean(running_losses)) if running_losses else 0.0
        episode_mean_losses.append(mean_loss)

        avg_last_20 = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
        print(
            f"Episode {episode + 1}/{max_episodes}  Steps: {steps}  Reward: {ep_reward:.2f}  "
            f"Avg(20): {avg_last_20:.2f}  Epsilon: {epsilon:.3f}  MeanLoss: {mean_loss:.4f}"
        )

        if (episode + 1) % 50 == 0:
            eval_rewards = []
            for _ in range(3):
                obs_eval, info_eval = env.reset()
                state_eval = flatten_obs(obs_eval)
                done_eval = False
                ep_r_eval = 0.0
                steps_eval = 0
                while not done_eval and steps_eval < 600:
                    with torch.no_grad():
                        s_eval = torch.tensor(state_eval, dtype=torch.float32, device=device).unsqueeze(0)
                        q_values_eval = policy_net(s_eval)
                        action_eval = int(torch.argmax(q_values_eval, dim=1).item())
                    next_obs_eval, r_eval, term_eval, trunc_eval, info_eval = env.step(action_eval)
                    state_eval = flatten_obs(next_obs_eval)
                    done_eval = term_eval or trunc_eval
                    ep_r_eval += r_eval
                    steps_eval += 1
                eval_rewards.append(ep_r_eval)
            print(f"Greedy eval mean reward over 3 episodes: {np.mean(eval_rewards):.2f}")

        # TensorBoard logging
        writer.add_scalar("Reward/Episode", ep_reward, episode)
        writer.add_scalar("Reward/Average_20", avg_last_20, episode)
        writer.add_scalar("Epsilon", epsilon, episode)
        writer.add_scalar("Loss/Mean", mean_loss, episode)
        writer.add_scalar("Steps", steps, episode)

    writer.close()
    env.close()

    save_path = os.path.abspath("dqn_highway.pt")
    print("Saving model to", save_path)
    torch.save(policy_net.state_dict(), save_path)
    print("Saved:", os.path.exists("dqn_highway.pt"))

    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN on highway-v0: Episode Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve.png")

    plt.figure()
    plt.plot(episode_mean_losses)
    plt.xlabel("Episode")
    plt.ylabel("Mean Loss")
    plt.title("DQN on highway-v0: Mean Loss per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")

    plt.figure()
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (steps)")
    plt.title("DQN on highway-v0: Episode Length")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("episode_lengths.png")

    print("Saved learning_curve.png, loss_curve.png, episode_lengths.png")

if __name__ == "__main__":
    print("CWD:", os.getcwd())
    train_dqn()
