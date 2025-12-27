# 2D Highway Reinforcement Learning Racing

---

## Project Overview
This project implements a reinforcement learning agent trained to navigate multi-lane highway traffic autonomously. The agent learns to maximize speed while avoiding collisions and efficiently overtaking slower vehicles through thousands of simulated driving episodes.

Starting with no prior knowledge, the agent develops a driving policy entirely through experience. The trained model demonstrates competent highway navigation, including strategic lane changes and safe overtaking maneuvers.

---

## Deep Q-Learning Implementation

The agent uses **Deep Q-Learning (DQN)**, combining neural networks with Q-learning to map states to optimal actions. The implementation includes several architectural improvements over the baseline DQN algorithm.

### Observation Space
The agent receives structured state information about the highway environment:
- Relative positions of nearby vehicles
- Vehicle velocities
- Lane assignments

This representation provides complete spatial awareness without requiring raw image processing.

### Reward Structure
The learning process is driven by a carefully tuned reward function:
- **Positive rewards**: Forward velocity, successful overtakes, collision avoidance
- **Negative rewards**: Collisions, low speeds, unsafe following distances

The neural network learns to maximize cumulative reward over time, developing strategies that balance speed with safety.

### Architectural Improvements

**Double DQN**  
Addresses the value overestimation problem inherent in standard DQN. By decoupling action selection from action evaluation using two networks, Double DQN produces more stable and accurate Q-value estimates. This prevents the agent from overvaluing risky actions based on limited experience.

**Dueling DQN**  
Separates the Q-value estimation into two components:
- **State value function V(s)**: Estimates the value of being in a particular state
- **Advantage function A(s,a)**: Estimates the relative value of each action in that state

This decomposition allows the network to learn state values independently from action-specific advantages, leading to faster convergence and more robust policies. In highway driving, this is particularly useful—the agent can learn that having open road ahead is valuable regardless of the immediate action taken.

![DQN Architecture Diagram](https://github.com/user-attachments/assets/9a598cb3-7b44-4884-a63a-a615129d1a20)

---

## Training Infrastructure

Training reinforcement learning agents is computationally intensive, requiring thousands of environment simulations and neural network updates. This project utilized the **UNC School System H200 GPU Cluster** to accelerate the training process.

The cluster enabled:
- Parallel episode simulations
- Faster neural network forward and backward passes
- Rapid iteration through hyperparameter configurations

This infrastructure reduced training time from days on consumer hardware to hours on the cluster.

---

## Development Iterations

### Challenge 1: Overly Conservative Behavior
Initial training produced an agent that prioritized collision avoidance to the extent that it drove well below optimal speeds. The reward structure inadvertently incentivized minimal risk-taking.

**Solution**: Modified the reward function to penalize low velocities and provide explicit bonuses for overtaking maneuvers. This forced the agent to find strategies that balanced speed with safety rather than optimizing purely for collision avoidance.

### Challenge 2: Unsafe Following Distances
After adjusting for speed, the agent began exhibiting aggressive behavior—tailgating and making risky lane changes with insufficient spacing.

**Solution**: Implemented a distance-based penalty that activates when the agent enters an unsafe following distance. This "safety bubble" encourages the agent to maintain appropriate spacing while still pursuing overtaking opportunities when safe.

### Challenge 3: Lane Change Commitment
The agent occasionally initiated lane changes but failed to complete them, resulting in oscillating behavior between lanes.

**Solution**: Refined the observation space to provide better visibility into adjacent lanes and restructured lane change rewards. The Dueling architecture's separation of state value from action advantage also helped the agent learn when to commit to or abort lane changes.

![Training progression visualization - placeholder]

---

## Results

Final model performance after training:

- **Success Rate**: ~90%
- **Average Distance**: ~1498 meters / 1500 meters
- **Collision Rate**: ~10%

The trained agent successfully navigates most highway scenarios, demonstrating effective lane change timing and overtaking judgment. Remaining failures typically occur in high-density traffic situations requiring precise gap assessment.

<img width="1159" height="330" alt="image" src="https://github.com/user-attachments/assets/b86a4782-0960-44d2-859b-f511ed7985cb" />

[Agent performance Visualization Video](https://drive.google.com/file/d/172xpY-mhqs6G1GQfsff50NMZ6r2tTi5g/view?usp=sharing)

---

## Presentation
[2D Highway Reinforcement Learning Presentation](https://docs.google.com/presentation/d/1QHXi0lgQYT10GyXixdnGkv2ifG6T5bXIf8xAtb5uVmk/edit?usp=sharing)

## Usage

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Simulation
Watch the trained agent drive:
```bash
python run_dqn_highway.py
```

### 3. Train (Optional)
To train the model from scratch:
```bash
python train_dqn_highway.py
```

## Credits
*   **Simulation Environment**: Thanks to **Edouard Leurent** for creating the [highway-env](https://github.com/eleurent/highway-env) library, which made this project possible.
