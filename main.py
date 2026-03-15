"""
Neural-X: Reinforcement Learning Driving Simulator
Main Entry Point

This script initializes the training engine and launches the performance 
dashboard for the highway navigation agent.

Architecture:
    - Trainer: Manages the RL loop and agent-environment interaction.
    - GameUI: Provides the high-fidelity visualization and telemetry dashboard.

Usage:
    python main.py
"""
from src.train import Trainer
from src.render import GameUI

def main():
    """Initializes and executes the Neural-X simulation suite."""
    trainer = Trainer()
    ui = GameUI(trainer)
    ui.run()

if __name__ == "__main__":
    main()
