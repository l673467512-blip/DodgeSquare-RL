# DodgeSquare: A Deep RL Survival Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Project Overview
[cite_start]**NeuralEvade** is a reinforcement learning project that trains an autonomous agent to dodge randomized obstacles in a continuous 2D environment[cite: 4, 5]. [cite_start]By implementing a **Deep Q-Network (DQN)**, the agent learns to interpret spatial coordinates and velocities to maximize its survival time[cite: 7, 42]. [cite_start]This project demonstrates the transition from stochastic exploration to sophisticated predictive dodging[cite: 50, 74].

## 2. Key Features
- [cite_start]**DQN Architecture**: Utilizes a multi-layer fully connected network (128-64-3) with ReLU activation for complex state-action mapping[cite: 43, 44].
- [cite_start]**Adaptive Epsilon-Greedy Strategy**: Balances exploration and exploitation via episodic decay, ensuring robust policy convergence[cite: 49, 53].
- [cite_start]**Advanced State Representation**: A 4-dimensional normalized vector consisting of Agent/Obstacle coordinates and velocities[cite: 17, 46].
- [cite_start]**Interactive UI**: Includes a comprehensive start menu, multi-life system (3 lives), and a Game Over state machine for both AI and manual play[cite: 16, 32].

## 3. Technology Stack
- [cite_start]**Language**: Python 3.8+ [cite: 8]
- [cite_start]**DL Framework**: PyTorch [cite: 9, 10]
- [cite_start]**Game Engine**: Pygame [cite: 10, 11]
- [cite_start]**Analysis**: Matplotlib (for learning curve visualization) [cite: 52, 73]

## 4. Performance Results
The following learning curve demonstrates the agent's performance over 200 training episodes.
![Learning Curve](./final_curve.png)

### Key Observations
1. [cite_start]**Stochastic Phase**: In early episodes, the agent moves randomly with a high Epsilon ($\epsilon = 1.0$), resulting in frequent collisions[cite: 50, 73].
2. [cite_start]**Convergence**: As the Epsilon decays, the agent relies more on its learned Q-values, leading to stable survival scores and higher average rewards[cite: 51, 74].
3. [cite_start]**Robustness**: The agent successfully learns to avoid obstacles even with randomized speeds, proving the effectiveness of the DQN model[cite: 71, 77].



## 5. Getting Started
### Installation
```bash
pip install -r requirements.txt
```

### Usage
To train the AI: Run python main.py and click the screen to start the process.
To play manually: Run python env.py to test physics, UI controls, and the Game Over state.

## 6. References
[1] Wang, X. (2025). TankWar: Tank battle game [Computer software]. GitHub.  
[2] IronSpiderMan. (2020). TankWar: Classic tank battle game implemented with Python and Pygame [Computer software]. GitHub.  
[3] te. (2025). Tetris-deep-Q-learning-pytorch [Computer software]. GitCode.  

