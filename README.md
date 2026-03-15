# 🏎️ Neural-X: Reinforcement Learning Driving Simulator

Neural-X is a high-fidelity driving simulation designed to explore the intersection of **Tabular Q-Learning** and real-time **Telemetric Visualization**. Built with Python and Pygame, it features a sophisticated autonomous agent capable of navigating a dynamic 3-lane highway while optimizing for safety and efficiency.

![Project Banner](https://img.shields.io/badge/Status-Research_Lab-00ffb4?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge)

---

## 🔬 Executive Summary

The primary objective of this project is to model **Operational Risk** in autonomous systems. Beyond simple collision avoidance, Neural-X quantifies decision-making confidence, analyzes lateral lane biases, and calculates Time-To-Collision (TTC) metrics in real-time.

### Core Systems:
*   **AI Analytics Dashboard**: A glassmorphism-inspired HUD displaying global rewards, survival trends, and risk density.
*   **Proximity Radar (Sonar)**: A multi-lane sensor array that discretizes the environment for the RL agent.
*   **Dynamic Risk Modeling**: An exponential decay risk engine that visualizes "danger auras" based on obstacle proximity.
*   **Automated Insights Engine**: A heuristic-based analyst that provides high-level architectural feedback on policy convergence and behavioral biases.

---

## 🏗️ Technical Architecture

The codebase follows a modular design pattern, separating environment physics from the reinforcement learning logic and the visualization layer.

| Module | Responsibility |
| :--- | :--- |
| **`agent.py`** | **The Brain**: Implements a Tabular Q-Learning algorithm with epsilon-greedy exploration and decision-confidence metrics. |
| **`env.py`** | **The World**: A 3-lane highway simulation with discretized state-spaces and a reward-shaping engine. |
| **`stats.py`** | **The Analyst**: Aggregates telemetry data and implements the Senior Insight generator for trend analysis. |
| **`render.py`** | **The Interface**: A Pygame-based rendering engine featuring modern HUD aesthetics and real-time charting. |
| **`train.py`** | **The Orchestrator**: Manages the training loop, knowledge propagation, and checkpointing. |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed.

### Installation

```bash
# Clone the repository
git clone https://github.com/Danezc/carro_ia.git
cd carro_ia

# Install dependencies
pip install -r requirements.txt
```

### Execution

```bash
python main.py
```

---

## 🎮 Operational Controls

Once the simulation is active, use the following hotkeys to interact with the system:

| Key | Operation | Description |
| :--- | :--- | :--- |
| **`P`** | **Toggle Autoplay** | Switches between manual inspection and autonomous navigation (Exploitation mode). |
| **`T`** | **Fast-Train (x50)** | Executes 50 training episodes in the background to accelerate learning. |
| **`R`** | **System Reset** | Reinitializes the environment state and resets current session episodic metrics. |
| **`ESC`** | **Exit Suite** | Safely terminates all processes and closes the dashboard. |

---

## 🏁 Future Roadmap

- [ ] **DQN Integration**: Migration from Tabular Q-Learning to Deep Q-Networks for handling continuous state spaces.
- [ ] **Multi-Agent Traffic**: Implementing complex traffic patterns where obstacles also have autonomous behaviors.
- [ ] **Advanced Telemetry**: Persistence of training sessions via SQLite for long-term policy analysis.

---

*Developed as a research initiative at the intersection of RL and Data Visualization.*
