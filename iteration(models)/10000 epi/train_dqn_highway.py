import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os


# This is the neural network that learns how to drive
class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        # Build the neural network layers - takes sensor data in and gives action scores out
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  # First layer: takes input and expands to 256 neurons
            nn.ReLU(),  # Activation function to make it non-linear
            nn.Linear(256, 256),  # Second layer: processes the information
            nn.ReLU(),  # Another activation
            nn.Linear(256, num_actions)  # Final layer: outputs score for each possible action
        )

    def forward(self, x):
        # Pass the input through the network to get action scores
        return self.net(x)


# This stores past experiences so the agent can learn from them later
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity  # Maximum number of experiences to store
        self.state_dim = state_dim  # Size of the state (sensor data)
        # Create arrays to store different parts of each experience
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)  # What the agent saw
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)  # What happened next
        self.actions = np.zeros((capacity,), dtype=np.int64)  # What action was taken
        self.rewards = np.zeros((capacity,), dtype=np.float32)  # How good/bad that action was
        self.dones = np.zeros((capacity,), dtype=np.float32)  # Whether the episode ended
        self.idx = 0  # Where to store the next experience
        self.size = 0  # How many experiences we currently have

    def add(self, state, action, reward, next_state, done):
        # Save a new experience (what happened at one step)
        self.states[self.idx] = state
        self.next_states[self.idx] = next_state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = float(done)
        # Move to next position, wrap around if buffer is full
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        # Randomly pick some past experiences to learn from
        idxs = np.random.randint(0, self.size, size=batch_size)
        # Convert to PyTorch tensors and move to GPU if available
        states = torch.tensor(self.states[idxs], dtype=torch.float32, device=device)
        next_states = torch.tensor(self.next_states[idxs], dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions[idxs], dtype=torch.int64, device=device)
        rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones[idxs], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

# This wrapper modifies the rewards to help the agent learn better driving behavior
class AdvancedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.vehicles_ahead = set()  # Track which cars are in front of us

    def reset(self, **kwargs):
        # Start a new episode
        obs, info = self.env.reset(**kwargs)
        self.vehicles_ahead = set()
        if hasattr(self.env.unwrapped, "vehicle") and hasattr(self.env.unwrapped, "road"):
            ego = self.env.unwrapped.vehicle  # Our car
            ego_x = ego.position[0]
            if hasattr(ego, "speed"):
                ego.speed = 51.5  # Set our car's speed to 51.5 m/s
            # Set all other cars to a fixed speed
            for v in self.env.unwrapped.road.vehicles:
                if v is not ego:
                    target = 35.0  # Other cars go at 35 m/s
                    if hasattr(v, "speed"):
                        v.speed = target
                    if v.position[0] > ego_x:  # If car is ahead of us
                        self.vehicles_ahead.add(v)

        return obs, info

    def step(self, action):
        # Take a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Keep our speed and other cars' speeds fixed
        if hasattr(self.env.unwrapped, "vehicle") and hasattr(self.env.unwrapped, "road"):
            ego = self.env.unwrapped.vehicle
            if hasattr(ego, "speed"):
                ego.speed = 51.5  # Our car stays at 51.5 m/s
            for v in self.env.unwrapped.road.vehicles:
                if v is ego:
                    continue
                target = 35.0  # Other cars stay at 35 m/s
                if hasattr(v, "speed"):
                    v.speed = target

        # Calculate our current speed
        ego_kin = obs[0]
        ego_vx = float(ego_kin[3])  # Velocity in x direction
        ego_vy = float(ego_kin[4])  # Velocity in y direction
        speed = float(np.sqrt(ego_vx ** 2 + ego_vy ** 2))  # Total speed

        # Find the closest car in front of us and check if any cars are ahead
        min_front_gap = 100.0
        cars_ahead_detected = False
        for i in range(1, len(obs)):
            if obs[i][0] == 0:  # Skip if no car detected
                continue
            rel_x = float(obs[i][1])  # Distance ahead (positive = ahead, negative = behind)
            rel_y = float(obs[i][2])  # Distance to the side
            if rel_x > 0:  # If car is ahead of us (in any lane)
                cars_ahead_detected = True  # At least one car is ahead
                if abs(rel_y) < 2.0:  # If car is in our lane
                    if rel_x < min_front_gap:
                        min_front_gap = rel_x

        # Give bonus reward for passing cars
        overtaking_bonus = 0.0
        if hasattr(self.env.unwrapped, "vehicle") and hasattr(self.env.unwrapped, "road"):
            raw_ego_x = self.env.unwrapped.vehicle.position[0]
            current_vehicles = set(self.env.unwrapped.road.vehicles)
            passed_vehicles = set()
            for v in list(self.vehicles_ahead):
                if v.position[0] < raw_ego_x:  # We passed this car!
                    overtaking_bonus += 1.0
                    passed_vehicles.add(v)
                if v not in current_vehicles:  # Car left the road
                    passed_vehicles.add(v)
            self.vehicles_ahead -= passed_vehicles
            # Update which cars are now ahead of us
            for v in current_vehicles:
                if v is not self.env.unwrapped.vehicle and v.position[0] > raw_ego_x:
                    self.vehicles_ahead.add(v)

        # Modify the reward based on how well the agent is driving
        shaped_reward = reward

        if not terminated:
            shaped_reward += 0.05  # Small bonus for not crashing

        # Penalize for being too close to cars ahead
        if min_front_gap < 5.0:
            shaped_reward -= 1.0  # Very close = bad
        elif min_front_gap < 15.0:
            shaped_reward -= 0.5 * (15.0 - min_front_gap) / 10.0  # Moderately close = slightly bad
        elif min_front_gap < 40.0:
            shaped_reward += 0.2 * (min_front_gap - 15.0) / 25.0  # Good distance = good
        else:
            shaped_reward += 0.2  # Very far = very good

        if action == 0 or action == 2:
            shaped_reward -= 0.05  # Small penalty for lane changes (to encourage smooth driving)

        # Reward for maintaining target speed
        v_target = 51.5
        if v_target > 0.0:
            speed_term = -0.5 * ((speed - v_target) / v_target) ** 2  # Penalize if speed is off
        else:
            speed_term = 0.0
        if min_front_gap < 20.0:
            speed_term *= 0.3  # Less important if we're close to a car
        shaped_reward += speed_term

        shaped_reward += overtaking_bonus * 1.0  # Bonus for passing cars

        # End episode early if no cars are ahead and we've traveled far enough
        # This saves time when there's nothing left to interact with
        if not terminated and hasattr(self.env.unwrapped, "vehicle") and hasattr(self.env.unwrapped, "road"):
            ego_x = self.env.unwrapped.vehicle.position[0]
            
            # Check if there are any vehicles ahead on the road
            vehicles_ahead_on_road = False
            for v in self.env.unwrapped.road.vehicles:
                if v is not self.env.unwrapped.vehicle and v.position[0] > ego_x:
                    vehicles_ahead_on_road = True
                    break
            
            # If no cars ahead in observation AND no vehicles ahead on road, end episode early
            # Lower threshold to 800m to end sooner when cars are gone
            if not cars_ahead_detected and not vehicles_ahead_on_road and ego_x > 800.0:
                truncated = True  # End the episode early

        return obs, shaped_reward, terminated, truncated, info

# Normalize observation values to make training easier
def normalize_obs(obs):
    obs = np.array(obs, dtype=np.float32)
    obs[..., 1] /= 100.0  # Normalize x position
    obs[..., 2] /= 10.0  # Normalize y position
    obs[..., 3] /= 55.0  # Normalize x velocity
    obs[..., 4] /= 55.0  # Normalize y velocity
    return obs

# Convert the 2D observation array into a 1D array for the neural network
def flatten_obs(obs):
    return normalize_obs(obs).reshape(-1)

# Create the highway driving environment with custom settings
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
        "vehicles_count": 50,  # Increased from 30 to make highway denser with more cars
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
    env = AdvancedRewardWrapper(env)
    return env



# Main training function - this is where the agent learns to drive
def train_dqn():
    env = make_env(render_mode=None)  # Create the highway environment
    obs, info = env.reset()  # Reset to start position
    state = flatten_obs(obs)  # Convert observation to format neural network can use
    state_dim = state.shape[0]  # How many numbers describe the current state
    num_actions = env.action_space.n  # How many actions we can take (left, right, forward, etc.)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    policy_net = DQN(state_dim, num_actions).to(device)  # Main network that makes decisions
    target_net = DQN(state_dim, num_actions).to(device)  # Target network for stable learning
    
    model_path = "dqn_highway.pt"  # Where to save the trained model
    start_episode = 0  # Which episode to start from (0 if new training)
    total_steps = 0  # Total number of steps taken across all episodes
    episode_rewards = []  # Store rewards for each episode
    episode_lengths = []  # Store how long each episode lasted
    episode_mean_losses = []  # Store average loss for each episode
    
    # Look for saved checkpoints to resume training
    checkpoint_files = []
    try:
        checkpoint_files = [f for f in os.listdir(".") if f.startswith("dqn_highway_checkpoint_ep") and f.endswith(".pt")]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split("_ep")[1].split(".")[0]), reverse=True)  # Sort by episode number
    except Exception as e:
        checkpoint_files = []
    
    # Check if a saved model exists
    model_exists = False
    try:
        if os.path.exists(model_path):
            model_exists = True
    except Exception as e:
        model_exists = False
    
    # If we have checkpoints but no main model, use the latest checkpoint
    if checkpoint_files and not model_exists:
        latest_checkpoint = checkpoint_files[0]
        checkpoint_episode = int(latest_checkpoint.split("_ep")[1].split(".")[0])
        model_path = latest_checkpoint
        model_exists = True
        start_episode = checkpoint_episode
    
    # Load saved model if it exists
    if model_exists:
        try:
            checkpoint = torch.load(model_path, map_location=device)  # Load the saved weights
            if "module." in str(list(checkpoint.keys())[0]):
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}  # Remove DataParallel prefix if needed
            policy_net.load_state_dict(checkpoint)  # Load weights into policy network
            target_net.load_state_dict(policy_net.state_dict())  # Copy to target network
            if "checkpoint" in model_path:
                checkpoint_episode = int(model_path.split("_ep")[1].split(".")[0])
                start_episode = checkpoint_episode
                total_steps = checkpoint_episode * 100  # Estimate total steps
            else:
                start_episode = 2000  # Assume main model was trained for 2000 episodes
                total_steps = 200000
        except Exception as e:
            print(f"Error loading model: {e}")
            target_net.load_state_dict(policy_net.state_dict())
    else:
        target_net.load_state_dict(policy_net.state_dict())  # Initialize target net from policy net
    
    target_net.eval()  # Set target network to evaluation mode (no training updates)

    optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)  # Optimizer to adjust network weights
    buffer = ReplayBuffer(capacity=120000, state_dim=state_dim)  # Memory to store past experiences
    writer = SummaryWriter(log_dir="runs/dqn_highway")  # For logging training progress

    # Training hyperparameters - these control how the agent learns
    gamma = 0.99  # Discount factor for future rewards (how much we care about future vs now)
    batch_size = 64  # How many experiences to learn from at once
    start_learning = 2000  # Wait this many steps before starting to learn
    target_update_freq = 2000  # Update target network every this many steps
    max_episodes = 10000  # Total number of episodes to train
    max_steps = 10000  # Maximum steps per episode

    # Epsilon-greedy exploration: start exploring randomly, gradually use learned policy more
    epsilon_start = 1.0  # Start with 100% random actions
    epsilon_end = 0.05  # End with 5% random actions
    epsilon_decay_steps = 200000  # Gradually decrease exploration over this many steps

    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    episode_mean_losses = []

    # Main training loop - run episodes one by one
    for episode in range(start_episode, start_episode + max_episodes):
        obs, info = env.reset()  # Start a new episode
        state = flatten_obs(obs)
        done = False
        ep_reward = 0.0  # Total reward for this episode
        steps = 0  # Steps taken in this episode
        running_losses = []  # Track losses during this episode

        # Run steps until episode ends or max steps reached
        while not done and steps < max_steps:
            # Calculate current exploration rate (epsilon)
            epsilon = max(
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end) * total_steps / epsilon_decay_steps
            )

            # Choose action: random (explore) or from network (exploit)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                with torch.no_grad():  # Don't calculate gradients for action selection
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(s)  # Get Q-values for each action
                    action = int(torch.argmax(q_values, dim=1).item())  # Choose best action

            # Take action in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = flatten_obs(next_obs)
            done_flag = terminated or truncated

            # Clip rewards to prevent extreme values
            clipped_reward = max(min(reward, 5.0), -5.0)
            buffer.add(state, action, clipped_reward, next_state, done_flag)  # Save experience

            state = next_state
            ep_reward += reward
            steps += 1
            total_steps += 1

            # Train the network if we have enough experiences
            if buffer.size >= start_learning and total_steps % 4 == 0:
                states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size, device)

                # Calculate Q-values for current states
                q_values = policy_net(states_b)
                q_values = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                # Calculate target Q-values using target network (for stability)
                with torch.no_grad():
                    next_actions = policy_net(next_states_b).argmax(1)  # Best action from policy net
                    next_q_values = target_net(next_states_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # Q-value from target net
                    targets = rewards_b + gamma * (1.0 - dones_b) * next_q_values  # Bellman equation

                # Calculate loss and update network
                loss = nn.MSELoss()(q_values, targets)

                optimizer.zero_grad()  # Clear old gradients
                loss.backward()  # Calculate new gradients
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)  # Prevent gradient explosion
                optimizer.step()  # Update weights

                running_losses.append(loss.item())

            # Periodically update target network to match policy network
            if total_steps % target_update_freq == 0 and total_steps > 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done_flag:
                break

        # Record episode statistics
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        mean_loss = float(np.mean(running_losses)) if running_losses else 0.0
        episode_mean_losses.append(mean_loss)

        # Calculate and print episode summary
        avg_last_20 = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
        print(
            f"Episode {episode + 1}/{start_episode + max_episodes}  Steps: {steps}  Reward: {ep_reward:.2f}  "
            f"Avg(20): {avg_last_20:.2f}  Epsilon: {epsilon:.3f}  MeanLoss: {mean_loss:.4f}"
        )

        # Save checkpoint every 2000 episodes
        if (episode + 1) % 2000 == 0:
            checkpoint_path = f"dqn_highway_checkpoint_ep{episode + 1}.pt"
            try:
                torch.save(policy_net.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

        # Evaluate agent performance every 50 episodes (no exploration, just best actions)
        if (episode + 1) % 50 == 0:
            eval_rewards = []
            for _ in range(3):  # Run 3 evaluation episodes
                obs_eval, info_eval = env.reset()
                state_eval = flatten_obs(obs_eval)
                done_eval = False
                ep_r_eval = 0.0
                steps_eval = 0
                while not done_eval and steps_eval < 600:
                    with torch.no_grad():  # No training during evaluation
                        s_eval = torch.tensor(state_eval, dtype=torch.float32, device=device).unsqueeze(0)
                        q_values_eval = policy_net(s_eval)
                        action_eval = int(torch.argmax(q_values_eval, dim=1).item())  # Always choose best action
                    next_obs_eval, r_eval, term_eval, trunc_eval, info_eval = env.step(action_eval)
                    state_eval = flatten_obs(next_obs_eval)
                    done_eval = term_eval or trunc_eval
                    ep_r_eval += r_eval
                    steps_eval += 1
                eval_rewards.append(ep_r_eval)
            avg_eval = np.mean(eval_rewards)
            writer.add_scalar("eval/mean_reward", avg_eval, episode + 1)  # Log to TensorBoard

        # Log metrics to TensorBoard for visualization
        writer.add_scalar("Reward/Episode", ep_reward, episode)
        writer.add_scalar("Reward/Average_20", avg_last_20, episode)
        writer.add_scalar("Epsilon", epsilon, episode)
        writer.add_scalar("Loss/Mean", mean_loss, episode)
        writer.add_scalar("Steps", steps, episode)

    writer.close()  # Close TensorBoard writer
    env.close()  # Close environment

    # Save final trained model
    save_path = model_path
    try:
        torch.save(policy_net.state_dict(), save_path)
    except Exception as e:
        print(f"Error saving model: {e}")
        save_path = "dqn_highway.pt"
        torch.save(policy_net.state_dict(), save_path)

    # Create plots showing how the agent improved over time
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN on highway-v0: Episode Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve.png")  # Save reward plot

    plt.figure()
    plt.plot(episode_mean_losses)
    plt.xlabel("Episode")
    plt.ylabel("Mean Loss")
    plt.title("DQN on highway-v0: Mean Loss per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")  # Save loss plot

    plt.figure()
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (steps)")
    plt.title("DQN on highway-v0: Episode Length")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("episode_lengths.png")  # Save episode length plot

if __name__ == "__main__":
    train_dqn()  # Start training when script is run
