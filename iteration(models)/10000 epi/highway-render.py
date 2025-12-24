import gymnasium as gym
import highway_env
import numpy as np

def make_env(render_mode="human"):
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["x", "y", "vx", "vy", "heading"],
            "absolute": True,
            "order": "sorted"
        },
        "policy_frequency": 15,
        "duration": 40,
        "vehicles_count": 15,
        "controlled_vehicles": 1,
        "lanes_count": 4,
        "reward_speed_range": [20, 30],
        "collision_reward": -5,
        "high_speed_reward": 0.5,
        "right_lane_reward": 0.1,
        "reward_speed_range_fraction": 1.0
    }

    env = gym.make(
        "highway-v0",
        render_mode=render_mode,
        config=config
    )
    return env

def run_random_policy(episodes=3, max_steps=500, render_mode="human"):
    env = make_env(render_mode=render_mode)
    for ep in range(episodes):
        state, info = env.reset()
        done = False
        step = 0
        ep_reward = 0.0
        while not done and step < max_steps:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            state = next_state
            step += 1
        print(f"Episode {ep + 1} finished. Steps: {step}, total reward: {ep_reward}")
    env.close()

if __name__ == "__main__":
    run_random_policy(episodes=5, max_steps=400, render_mode="human")
