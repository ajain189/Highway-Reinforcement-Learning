import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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


class replayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
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
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size, device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        states = torch.tensor(self.states[idxs], dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions[idxs], dtype=torch.int64, device=device)
        rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32, device=device)
        next_states = torch.tensor(self.next_states[idxs], dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones[idxs], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

class rewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.vehicles_ahead = []

    def reset(self):
        obs, info = self.env.reset()
        self.vehicles_ahead = []
        if hasattr(self.env.unwrapped, "vehicle") and hasattr(self.env.unwrapped, "road"):
            ego = self.env.unwrapped.vehicle
            ego_x = ego.position[0]
            if hasattr(ego, "speed"):
                ego.speed = 51.5
            for v in self.env.unwrapped.road.vehicles:
                if v is not ego and hasattr(v, "speed"):
                    v.speed = 35.0
                    if v.position[0] > ego_x:
                        self.vehicles_ahead.append(v)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if hasattr(self.env.unwrapped, "vehicle") and hasattr(self.env.unwrapped, "road"):
            ego = self.env.unwrapped.vehicle
            if hasattr(ego, "speed"):
                ego.speed = 51.5
            for v in self.env.unwrapped.road.vehicles:
                if v is not ego and hasattr(v, "speed"):
                    v.speed = 35.0

        ego_kin = obs[0]
        ego_vx = float(ego_kin[3])
        ego_vy = float(ego_kin[4])
        speed = float(np.sqrt(ego_vx * ego_vx + ego_vy * ego_vy))

        min_front_gap = 100.0
        cars_ahead_detected = False
        for i in range(1, len(obs)):
            if obs[i][0] == 0:
                continue
            rel_x = float(obs[i][1])
            rel_y = float(obs[i][2])
            if rel_x > 0:
                cars_ahead_detected = True
                if abs(rel_y) < 2.0:
                    if rel_x < min_front_gap:
                        min_front_gap = rel_x

        overtaking_bonus = 0.0
        if hasattr(self.env.unwrapped, "vehicle") and hasattr(self.env.unwrapped, "road"):
            raw_ego_x = self.env.unwrapped.vehicle.position[0]
            current_vehicles = list(self.env.unwrapped.road.vehicles)
            
            passed_vehicles = []
            for v in self.vehicles_ahead:
                if v.position[0] < raw_ego_x:
                    overtaking_bonus += 1.0
                    passed_vehicles.append(v)
                elif v not in current_vehicles:
                    passed_vehicles.append(v)
            
            self.vehicles_ahead = [v for v in self.vehicles_ahead if v not in passed_vehicles]
            self.vehicles_ahead.extend([v for v in current_vehicles if v is not self.env.unwrapped.vehicle and v.position[0] > raw_ego_x])

        shaped_reward = reward
        if not terminated:
            shaped_reward += 0.05

        if min_front_gap < 5.0:
            shaped_reward -= 1.0
        elif min_front_gap < 15.0:
            shaped_reward -= 0.5 * (15.0 - min_front_gap) / 10.0
        elif min_front_gap < 40.0:
            shaped_reward += 0.2 * (min_front_gap - 15.0) / 25.0
        else:
            shaped_reward += 0.2

        if action == 0 or action == 2:
            shaped_reward -= 0.05

        v_target = 51.5
        speed_term = -0.5 * ((speed - v_target) / v_target) ** 2
        if min_front_gap < 20.0:
            speed_term *= 0.3
        shaped_reward += speed_term
        shaped_reward += overtaking_bonus

        if not terminated and hasattr(self.env.unwrapped, "vehicle") and hasattr(self.env.unwrapped, "road"):
            ego_x = self.env.unwrapped.vehicle.position[0]
            vehicles_ahead_on_road = False
            for v in self.env.unwrapped.road.vehicles:
                if v is not self.env.unwrapped.vehicle and v.position[0] > ego_x:
                    vehicles_ahead_on_road = True
                    break
            if not cars_ahead_detected and not vehicles_ahead_on_road and ego_x > 800.0:
                truncated = True

        return obs, shaped_reward, terminated, truncated, info

def normalize_obs(obs):
    obs = np.array(obs, dtype=np.float32)
    for i in range(len(obs)):
        obs[i][1] = obs[i][1] / 100.0
        obs[i][2] = obs[i][2] / 10.0
        obs[i][3] = obs[i][3] / 55.0
        obs[i][4] = obs[i][4] / 55.0
    return obs

def flatten_obs(obs):
    obs_normalized = normalize_obs(obs)
    obs_flat = obs_normalized.reshape(obs_normalized.size)
    return obs_flat

def make_env(render_mode=None):
    config = {
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": False,
            "target_speeds": [50, 53]
        },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
            "order": "sorted",
            "normalize": False
        },
        "policy_frequency": 15,
        "duration": 60,
        "vehicles_count": 50,
        "controlled_vehicles": 1,
        "lanes_count": 4,
        "right_lane_reward": 0.0,
        "lane_change_reward": 0.0,
        "collision_reward": -40.0,
        "high_speed_reward": 0.0,
        "reward_speed_range": [20, 55],
        "speed_limit": 55,
        "normalize_reward": False
    }
    env = gym.make("highway-v0", config=config, render_mode=render_mode)
    env = rewardWrapper(env)
    return env

def get_episode_number(filename):
    parts = filename.split("_ep")
    if len(parts) > 1:
        number_part = parts[1].split(".")[0]
        return int(number_part)
    return 0

def load_checkpoint(model_path, device):
    if not os.path.exists(model_path):
        return None, 0
    checkpoint = torch.load(model_path, map_location=device)
    if list(checkpoint.keys())[0].startswith("module."):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    start_episode = get_episode_number(model_path) if "checkpoint" in model_path else 2000
    return checkpoint, start_episode

def train_dqn():
    env = make_env()
    obs, info = env.reset()
    state_dim = flatten_obs(obs).shape[0]
    num_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(state_dim, num_actions).to(device)
    target_net = DQN(state_dim, num_actions).to(device)
    
    checkpoint_files = sorted(
        [f for f in os.listdir(".") if f.startswith("dqn_highway_checkpoint_ep") and f.endswith(".pt")],
        key=get_episode_number,
        reverse=True
    )
    
    if os.path.exists("dqn_highway.pt"):
        checkpoint, start_episode = load_checkpoint("dqn_highway.pt", device)
    elif len(checkpoint_files) > 0:
        checkpoint, start_episode = load_checkpoint(checkpoint_files[0], device)
    else:
        checkpoint = None
        start_episode = 0
    
    if checkpoint:
        policy_net.load_state_dict(checkpoint)
    target_net.load_state_dict(policy_net.state_dict())
    
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
    buffer = replayBuffer(120000, state_dim)
    writer = SummaryWriter(log_dir="runs/dqn_highway")

    gamma = 0.99
    batch_size = 64
    start_learning = 2000
    target_update_freq = 2000
    max_episodes = 10000
    max_steps = 10000
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 200000

    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    episode_mean_losses = []

    for episode in range(start_episode, start_episode + max_episodes):
        obs, info = env.reset()
        state = flatten_obs(obs)
        done = False
        ep_reward = 0.0
        steps = 0
        running_losses = []

        while not done and steps < max_steps:
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * total_steps / epsilon_decay_steps
            if epsilon < epsilon_end:
                epsilon = epsilon_end

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_array = np.array(state, dtype=np.float32)
                    state_tensor = torch.tensor(state_array, dtype=torch.float32, device=device)
                    state_tensor = state_tensor.unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action_value = q_values.argmax()
                    action = int(action_value.item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = flatten_obs(next_obs)
            done = terminated or truncated

            clipped_reward = np.clip(reward, -5.0, 5.0)
            
            buffer.add(state, action, clipped_reward, next_state, done)
            
            state = next_state
            ep_reward += reward
            steps += 1
            total_steps += 1

            if buffer.size >= start_learning and total_steps % 4 == 0:
                states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size, device)
                q_values = policy_net(states_b)
                actions_b_expanded = actions_b.unsqueeze(1)
                q_values_selected = q_values.gather(1, actions_b_expanded)
                q_values_selected = q_values_selected.squeeze(1)

                with torch.no_grad():
                    next_actions = policy_net(next_states_b).argmax(1)
                    next_actions_expanded = next_actions.unsqueeze(1)
                    next_q_values = target_net(next_states_b)
                    next_q_values_selected = next_q_values.gather(1, next_actions_expanded)
                    next_q_values_selected = next_q_values_selected.squeeze(1)
                    targets = rewards_b + gamma * (1.0 - dones_b) * next_q_values_selected

                loss = nn.MSELoss()(q_values_selected, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()
                running_losses.append(loss.item())

            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        mean_loss = np.mean(running_losses) if running_losses else 0.0
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        episode_mean_losses.append(mean_loss)

        avg_last_20 = np.mean(episode_rewards[-20:])
        
        print(f"Episode {episode + 1}/{start_episode + max_episodes}  Steps: {steps}  Reward: {ep_reward:.2f}  Avg(20): {avg_last_20:.2f}  Epsilon: {epsilon:.3f}  MeanLoss: {mean_loss:.4f}")

        if (episode + 1) % 2000 == 0:
            checkpoint_path = f"dqn_highway_checkpoint_ep{episode + 1}.pt"
            torch.save(policy_net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        if (episode + 1) % 50 == 0:
            eval_rewards = []
            for _ in range(3):
                obs, _ = env.reset()
                state = flatten_obs(obs)
                done = False
                ep_reward = 0.0
                for _ in range(600):
                    if done:
                        break
                    with torch.no_grad():
                        action = int(policy_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax().item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    state = flatten_obs(obs)
                    done = terminated or truncated
                    ep_reward += reward
                eval_rewards.append(ep_reward)
            writer.add_scalar("eval/mean_reward", np.mean(eval_rewards), episode + 1)

    writer.close()
    env.close()
    torch.save(policy_net.state_dict(), "dqn_highway.pt")

if __name__ == "__main__":
    train_dqn()
