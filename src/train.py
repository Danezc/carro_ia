from dataclasses import dataclass
from typing import List, Dict, Any
from src.env import LaneEnv
from src.agent import QLearningAgent
from src.stats import LiveStats

@dataclass
class EpisodeRecord:
    steps: List[Dict[str, Any]]
    total_reward: float
    distance: int

class Trainer:
    def __init__(self):
        self.env = LaneEnv(horizon=12, spawn_prob=0.35, seed=7)
        self.agent = QLearningAgent(alpha=0.20, gamma=0.95, epsilon_decay=0.990, seed=7)
        self.stats = LiveStats(window=30)

        self.episodes = []
        self.best_idx = None
    
    def _update_best(self):
        if not self.episodes:
            return
        # Simple best: max distance
        best_dist = -1
        idx = -1
        for i, ep in enumerate(self.episodes):
            if ep.distance > best_dist:
                best_dist = ep.distance
                idx = i
        self.best_idx = idx

    def train(self, n_episodes: int = 200, keep_every: int = 50):
        for ep in range(n_episodes):
            s = self.env.reset()
            total = 0.0
            record_steps = []
            done = False
            crashed = False

            while not done:
                a = self.agent.act(s, training=True)
                s2, r, done, info = self.env.step(a)
                self.agent.learn(s, a, r, s2, done)
                total += r
                crashed = info["crashed"]

                record_steps.append({
                    "car_lane": self.env.car_lane,
                    "obstacles": [(ob.lane, ob.y) for ob in self.env.obstacles],
                    "action": a,
                    "reward": r,
                    "crashed": crashed,
                })
                s = s2

            self.agent.decay()

            # estadística para gráficas (sin mostrar bankroll)
            self.stats.add_episode(distance=info["distance"], total_reward=total, crashed=crashed, epsilon=self.agent.epsilon)

            if (ep % keep_every) == 0 or ep == n_episodes - 1:
                rec = EpisodeRecord(steps=record_steps, total_reward=total, distance=info["distance"])
                self.episodes.append(rec)
                self._update_best()
