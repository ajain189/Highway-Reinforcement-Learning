# 2D Highway Reinforcement Learning Racing

---

## Project Overview
This project explores **Reinforcement Learning (RL)** by training an autonomous agent to navigate a multi-lane highway. The goal is to drive as fast as possible while avoiding collisions and efficiently overtaking slower traffic.

## How It Works
The agent uses **Deep Q-Learning (DQN)** to make decisions. It observes the road (vehicle positions, speeds) and decides whether to change lanes or maintain its current path. Through training, it learns a policy that maximizes rewards for safe and efficient driving.

### Technical Implementation
The core algorithm is a Deep Q-Network (DQN), with some improvements
*   **Double DQN**: Improves stability by reducing the overestimation of action values.
*   **Dueling DQN**: Separates the estimation of state values and action advantages, leading to smoother driving behavior.
<img width="1225" height="685" alt="image" src="https://github.com/user-attachments/assets/9a598cb3-7b44-4884-a63a-a615129d1a20" />


## Training Hardware
Training deep RL models is pretty computationally intensive, so this model was trained using the **UNC School System H200 GPU Cluster**, which accelerated the simulation of thousands of driving episodes.

## Challenges & Iterations
*   **Passive Driving**: Initially, the agent would drive too slowly to avoid risk. I adjusted the reward function to penalize low speeds and incentivize overtaking.
*   **Safety**: To prevent tailgating, I implemented a "safety bubble" penalty that discourages the agent from getting too close to other vehicles.

## Results
*   **Success Rate**: ~90%
*   **Average Distance**: ~1498 meters(out of 1500)
*   **Collision Rate**: ~10%

# Presentation
*  [2D Highway Reinforcement Learning Presentation]([https://www.example.com](https://docs.google.com/presentation/d/1QHXi0lgQYT10GyXixdnGkv2ifG6T5bXIf8xAtb5uVmk/edit?usp=sharing))

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
