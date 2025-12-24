import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import time

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_size=512):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        
        if hidden_size >= 2048:
            num_layers = 6
        elif hidden_size >= 1024:
            num_layers = 5
        else:
            num_layers = 3
        
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, hidden_size // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size // 2, num_actions))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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
        for i in range(1, len(obs)):
            if obs[i][0] == 0:
                continue
            rel_x = float(obs[i][1])
            rel_y = float(obs[i][2])
            if rel_x > 0 and abs(rel_y) < 2.0:
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
        "duration": 1e9,
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

def load_model(state_dim, n_actions, device):
    checkpoint = torch.load("dqn_highway.pt", map_location=device)
    checkpoint_keys = list(checkpoint.keys())
    
    if checkpoint_keys[0].startswith("module."):
        checkpoint_clean = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    else:
        checkpoint_clean = checkpoint
    
    if len(checkpoint_keys) == 6:
        try:
            net0_shape = checkpoint_clean['net.0.weight'].shape
            net4_shape = checkpoint_clean['net.4.weight'].shape
            if len(net0_shape) == 2 and net0_shape[1] == state_dim and net4_shape[0] == n_actions:
                class OldDQN(nn.Module):
                    def __init__(self, input_dim, num_actions, hidden_size):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(input_dim, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, num_actions)
                        )
                    def forward(self, x):
                        return self.net(x)
                policy_net = OldDQN(state_dim, n_actions, net0_shape[0]).to(device)
                policy_net.load_state_dict(checkpoint_clean, strict=True)
                return policy_net
        except (KeyError, RuntimeError, ValueError):
            pass
    
    hidden_sizes_to_try = [4096, 3072, 2560, 2048, 1024, 512, 256]
    for hidden_size in hidden_sizes_to_try:
        try:
            policy_net = DQN(state_dim, n_actions, hidden_size=hidden_size).to(device)
            policy_net.load_state_dict(checkpoint_clean, strict=False)
            return policy_net
        except (KeyError, RuntimeError, ValueError):
            continue
    return None

def watch_agent():
    env = make_env(render_mode="human")
    obs, info = env.reset()
    state_dim = flatten_obs(obs).shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        policy_net = load_model(state_dim, n_actions, device)
        if policy_net is None:
            print("Error: Could not load model.")
            return
    except FileNotFoundError:
        print("No model found! Train first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    policy_net.eval()
    n_episodes = 10
    GOAL_DISTANCE = 1500.0
    successes = 0
    crashes = 0
    total_reward = 0
    total_distance = 0.0
    total_completion = 0.0

    print(f"Goal: Travel {GOAL_DISTANCE} meters without crashing.")

    for i in range(n_episodes):
        obs, info = env.reset()
        start_x = env.unwrapped.vehicle.position[0] if hasattr(env.unwrapped, "vehicle") else 0.0
        
        done = False
        episode_reward = 0
        distance_traveled = 0.0

        while not done and distance_traveled < GOAL_DISTANCE:
            with torch.no_grad():
                state_tensor = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=device).unsqueeze(0)
                action = int(policy_net(state_tensor).argmax().item())

            next_obs, reward, done, truncated, info = env.step(action)
            obs = next_obs
            episode_reward += reward

            if hasattr(env.unwrapped, "vehicle"):
                distance_traveled = env.unwrapped.vehicle.position[0] - start_x

            env.render()
            time.sleep(0.012)

        percent_complete = min(100.0, (distance_traveled / GOAL_DISTANCE) * 100.0)
        
        total_distance += distance_traveled
        total_completion += percent_complete
        total_reward += episode_reward

        if distance_traveled >= GOAL_DISTANCE:
            successes += 1
            print(f"Episode {i+1}: Success. Reached {distance_traveled:.1f}m. Reward: {episode_reward:.2f}")
        elif done:
            crashes += 1
            print(f"Episode {i+1}: Failed (Crash). Dist: {distance_traveled:.1f}m ({percent_complete:.1f}%). Reward: {episode_reward:.2f}")
        else:
            print(f"Episode {i+1}: Failed (Unknown). Dist: {distance_traveled:.1f}m ({percent_complete:.1f}%). Reward: {episode_reward:.2f}")

    accuracy = (successes / n_episodes) * 100
    avg_reward = total_reward / n_episodes
    avg_distance = total_distance / n_episodes
    avg_completion = total_completion / n_episodes

    print("\nEVALUATION REPORT")
    print(f"Goal Success Rate:   {accuracy:.1f}%")
    print(f"Average Distance:    {avg_distance:.1f} m")
    print(f"Avg Goal Completion: {avg_completion:.1f}%")
    print(f"Average Reward:      {avg_reward:.2f}")
    print(f"Crashes:             {crashes}/{n_episodes}")

    env.close()

if __name__ == "__main__":
    watch_agent()
