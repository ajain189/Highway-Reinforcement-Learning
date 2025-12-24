import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn

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

def make_env(render_mode="human"):
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

def watch_agent(model_path="dqn_highway.pt", max_steps=10000, episodes=None):
    env = make_env(render_mode="human")
    obs, info = env.reset()
    state = flatten_obs(obs)
    state_dim = state.shape[0]
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(state_dim, num_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    episode_index = 0

    try:
        while True:
            if episodes is not None and episode_index >= episodes:
                break

            obs, info = env.reset()
            state = flatten_obs(obs)
            done = False
            ep_reward = 0.0
            steps = 0
            episode_index += 1

            while not done and steps < max_steps:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(s)
                    action = int(torch.argmax(q_values, dim=1).item())

                obs, reward, terminated, truncated, info = env.step(action)
                state = flatten_obs(obs)
                done = terminated or truncated
                ep_reward += reward
                env.render()
                steps += 1

            print(f"Eval episode {episode_index} steps {steps} reward {ep_reward:.2f}")
    except KeyboardInterrupt:
        print("Stopping viewer")
    finally:
        env.close()

if __name__ == "__main__":
    watch_agent()
