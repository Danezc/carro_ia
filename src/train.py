from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from src.env import LaneEnv
from src.agent import QLearningAgent
from src.stats import LiveStats

@dataclass
class EpisodeRecord:
    """
    Data structure containing the step-by-step history of a training episode.
    
    Attributes:
        steps (List[Dict[str, Any]]): Sequential states, actions, and rewards.
        total_reward (float): Cumulative reward signal for the episode.
        distance (int): Total steps survived.
    """
    steps: List[Dict[str, Any]]
    total_reward: float
    distance: int

class Trainer:
    """
    Orchestration layer for training the RL Agent.
    
    Handles the interaction between the environment and the agent, manages
    the training loop, and aggregates performance statistics.

    Attributes:
        env (LaneEnv): The highway simulation environment.
        agent (QLearningAgent): The Q-Learning model being optimized.
        stats (LiveStats): Historical telemetry and analytics engine.
        episodes (List[EpisodeRecord]): Saved history of key training episodes.
    """
    def __init__(self):
        """Initializes the trainer with default environment and agent configurations."""
        self.env = LaneEnv(horizon=12, spawn_prob=0.35, seed=7)
        self.agent = QLearningAgent(alpha=0.20, gamma=0.95, epsilon_decay=0.990, seed=7)
        self.stats = LiveStats(window=30)

        self.episodes: List[EpisodeRecord] = []
        self.best_idx: Optional[int] = None
    
    def _update_best(self):
        """Identifies and updates the index of the best performing episode (max distance)."""
        if not self.episodes:
            return
            
        best_dist = -1
        idx = -1
        for i, ep in enumerate(self.episodes):
            if ep.distance > best_dist:
                best_dist = ep.distance
                idx = i
        self.best_idx = idx

    def train(self, n_episodes: int = 200, keep_every: int = 50):
        """
        Executes the training loop for a specified number of episodes.

        Args:
            n_episodes (int): Number of episodes to simulate. Defaults to 200.
            keep_every (int): Frequency of saving detailed episode records. Defaults to 50.
        """
        for ep in range(n_episodes):
            s = self.env.reset()
            total_reward = 0.0
            record_steps = []
            done = False
            crashed = False
            
            # Metrics for this specific episode
            risks = []
            ttcs = []
            
            while not done:
                # 1. Action selection (Epsilon-greedy)
                a = self.agent.act(s, training=True)
                
                # 2. Environment transition
                s2, r, done, info = self.env.step(a)
                
                # 3. Knowledge update (Bellman equation)
                self.agent.learn(s, a, r, s2, done)
                
                # 4. Telemetry collection
                total_reward += r
                crashed = info["crashed"]
                risks.append(info["risk"])
                ttcs.append(info["min_ttc"])

                record_steps.append({
                    "car_lane": self.env.car_lane,
                    "obstacles": [(ob.lane, ob.y) for ob in self.env.obstacles],
                    "action": a,
                    "reward": r,
                    "crashed": crashed,
                })
                s = s2

            # Post-episode: Exploration decay
            self.agent.decay()

            # Process aggregated telemetry
            avg_risk = sum(risks) / len(risks) if risks else 0
            avg_ttc = sum(ttcs) / len(ttcs) if ttcs else 0
            
            self.stats.add_episode(
                distance=info["distance"], 
                total_reward=total_reward, 
                crashed=crashed, 
                epsilon=self.agent.epsilon,
                avg_risk=avg_risk,
                avg_ttc=avg_ttc,
                final_lane_hist=self.env.lane_history
            )

            # Checkpoint saving
            if (ep % keep_every) == 0 or ep == n_episodes - 1:
                rec = EpisodeRecord(steps=record_steps, total_reward=total_reward, distance=info["distance"])
                self.episodes.append(rec)
                self._update_best()
