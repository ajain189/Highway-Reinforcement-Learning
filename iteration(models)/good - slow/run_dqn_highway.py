import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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
        if hasattr(self.env.unwrapped, "vehicle") and hasattr(self.env.unwrapped, "road"):
            ego = self.env.unwrapped.vehicle
            ego_x = ego.position[0]
            target = 35.0
            for v in self.env.unwrapped.road.vehicles:
                if v is ego:
                    continue
                if hasattr(v, "speed"):
                    v.speed = target
                if v.position[0] > ego_x:
                    self.vehicles_ahead.add(v)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if hasattr(self.env.unwrapped, "vehicle") and hasattr(self.env.unwrapped, "road"):
            ego = self.env.unwrapped.vehicle
            for v in self.env.unwrapped.road.vehicles:
                if v is ego:
                    continue
                target = 35.0
                if hasattr(v, "speed"):
                    v.speed = target

        ego_kin = obs[0]
        ego_vx = float(ego_kin[3])
        ego_vy = float(ego_kin[4])
        speed = float(np.sqrt(ego_vx ** 2 + ego_vy ** 2))

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
            current_vehicles = set(self.env.unwrapped.road.vehicles)
            passed_vehicles = set()
            for v in list(self.vehicles_ahead):
                if v.position[0] < raw_ego_x:
                    overtaking_bonus += 1.0
                    passed_vehicles.add(v)
                if v not in current_vehicles:
                    passed_vehicles.add(v)
            self.vehicles_ahead -= passed_vehicles
            for v in current_vehicles:
                if v is not self.env.unwrapped.vehicle and v.position[0] > raw_ego_x:
                    self.vehicles_ahead.add(v)

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

        v_target = 42.0
        if v_target > 0.0:
            speed_term = -0.5 * ((speed - v_target) / v_target) ** 2
        else:
            speed_term = 0.0
        if min_front_gap < 20.0:
            speed_term *= 0.3
        shaped_reward += speed_term

        shaped_reward += overtaking_bonus * 1.0

        return obs, shaped_reward, terminated, truncated, info

def normalize_obs(obs):
    obs = np.array(obs, dtype=np.float32)
    obs[..., 1] /= 100.0
    obs[..., 2] /= 10.0
    obs[..., 3] /= 43.0
    obs[..., 4] /= 43.0
    return obs

def flatten_obs(obs):
    return normalize_obs(obs).reshape(-1)

def make_env(render_mode=None):
    config = {
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "target_speeds": [41, 43]
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
        "vehicles_count": 30,
        "controlled_vehicles": 1,
        "lanes_count": 4,
        "right_lane_reward": 0.0,
        "lane_change_reward": 0.0,
        "collision_reward": -40.0,
        "high_speed_reward": 0.0,
        "reward_speed_range": [20, 45],
        "speed_limit": 45,
        "normalize_reward": False
    }
    env = gym.make("highway-v0", config=config, render_mode=render_mode)
    env = AdvancedRewardWrapper(env)
    return env

def watch_agent():
    env = make_env(render_mode="human")
    obs, info = env.reset()
    state_dim = flatten_obs(obs).shape[0]
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_dim, n_actions).to(device)

    try:
        policy_net.load_state_dict(torch.load("dqn_highway.pt", map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No model found! Train first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to architecture mismatch (DQN vs DuelingDQN or Stack Size). Retrain!")
        return
    policy_net.eval()

    n_episodes = 10
    GOAL_DISTANCE = 1500.0

    successes = 0
    crashes = 0
    total_reward = 0
    total_distance = 0.0
    total_completion = 0.0

    print(f"\nStarting Evaluation over {n_episodes} episodes...")
    print(f"GOAL: Travel {GOAL_DISTANCE} meters without crashing.")

    for i in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0

        start_x = 0.0
        if hasattr(env.unwrapped, "vehicle"):
            start_x = env.unwrapped.vehicle.position[0]

        distance_traveled = 0.0

        while not (done or truncated):
            if random.random() < 0.0:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(obs_tensor)
                    action = q_values.argmax().item()

            next_obs, reward, done, truncated, info = env.step(action)
            obs = next_obs
            episode_reward += reward
            steps += 1

            if hasattr(env.unwrapped, "vehicle"):
                current_x = env.unwrapped.vehicle.position[0]
                distance_traveled = current_x - start_x

            if hasattr(env.unwrapped, "vehicle"):
                current_distance = env.unwrapped.vehicle.position[0]
            else:
                current_distance = 0

            env.render()

            try:
                import pygame
                if hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
                    screen = env.unwrapped.viewer.screen
                    font = pygame.font.Font(None, 36)
                    text = font.render(f"Distance: {current_distance:.1f}m", True, (255, 255, 255))
                    text_rect = text.get_rect()
                    text_rect.topleft = (10, 10)
                    pygame.draw.rect(screen, (0, 0, 0), text_rect.inflate(10, 10))
                    screen.blit(text, text_rect)
                    pygame.display.flip()
            except:
                pass

        percent_complete = min(100.0, (distance_traveled / GOAL_DISTANCE) * 100.0)
        total_distance += distance_traveled
        total_completion += percent_complete
        total_reward += episode_reward

        if distance_traveled >= GOAL_DISTANCE:
            successes += 1
            print(f"Episode {i+1}: SUCCESS! Reached {distance_traveled:.1f}m. Reward: {episode_reward:.2f}")
        elif steps < 900:
            crashes += 1
            print(f"Episode {i+1}: FAILED (Crash). Dist: {distance_traveled:.1f}m ({percent_complete:.1f}%). Reward: {episode_reward:.2f}")
        else:
            print(f"Episode {i+1}: FAILED (Time). Dist: {distance_traveled:.1f}m ({percent_complete:.1f}%). Reward: {episode_reward:.2f}")

    accuracy = (successes / n_episodes) * 100.0
    avg_reward = total_reward / n_episodes
    avg_distance = total_distance / n_episodes
    avg_completion = total_completion / n_episodes

    print("\n" + "=" * 30)
    print("EVALUATION REPORT")
    print("=" * 30)
    print(f"Goal Success Rate:   {accuracy:.1f}%")
    print(f"Average Distance:    {avg_distance:.1f} m")
    print(f"Avg Goal Completion: {avg_completion:.1f}%")
    print(f"Average Reward:      {avg_reward:.2f}")
    print(f"Crashes:             {crashes}/{n_episodes}")
    print("=" * 30 + "\n")

    env.close()

if __name__ == "__main__":
    watch_agent()
