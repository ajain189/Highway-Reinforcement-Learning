import gymnasium as gym
import highway_env
import time  # To control simulation speed

def make_env(render_mode="human", policy_frequency=15, duration=60, vehicles_count=15, speed_limit=95.0, controlled_vehicle_speed=None):
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": vehicles_count,
            "features": ["x", "y", "vx", "vy", "heading"],
            "absolute": True, 
            "order": "sorted"
        },
        "policy_frequency": policy_frequency,
        "duration": duration,
        "vehicles_count": vehicles_count,
        "controlled_vehicles": 1,
        "lanes_count": 4,
        "reward_speed_range": [0, speed_limit],
        "speed_limit": speed_limit
    }
    env = gym.make("highway-v0", render_mode=render_mode, config=config)
    if controlled_vehicle_speed is not None:
        class FixedSpeedWrapper(gym.Wrapper):
            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                try:
                    ego = self.env.unwrapped.vehicle
                    if hasattr(ego, "speed"):
                        ego.speed = controlled_vehicle_speed
                except Exception:
                    pass
                return obs, info
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                try:
                    ego = self.env.unwrapped.vehicle
                    if hasattr(ego, "speed"):
                        ego.speed = controlled_vehicle_speed
                except Exception:
                    pass
                return obs, reward, terminated, truncated, info
        env = FixedSpeedWrapper(env)
    return env

def play_highway(max_episodes=5, max_steps=10000, frame_sleep=0.01, 
                 speed_limit=55.0, controlled_vehicle_speed=None):
    env = make_env(render_mode="human", duration=80, speed_limit=speed_limit,
                   controlled_vehicle_speed=controlled_vehicle_speed)
    for ep in range(max_episodes):
        obs, _ = env.reset()
        done, step = False, 0
        print(f"Episode {ep+1}: Play using the window, close it to stop this episode.")
        while not done and step < max_steps:
            action = None
            result = env.step(action)
            obs, _, terminated, truncated, _ = result
            done = terminated or truncated
            step += 1
            if frame_sleep > 0:
                time.sleep(frame_sleep)
        print(f"Episode {ep+1}: Steps={step}")
    env.close()

if __name__ == "__main__":
    play_highway(speed_limit=45.0)
