from dataclasses import dataclass, field
from collections import deque

@dataclass
class LiveStats:
    distances: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    crashes: list[int] = field(default_factory=list)  # 1 crash, 0 no
    epsilons: list[float] = field(default_factory=list)

    window: int = 30
    _recent_crashes: deque = field(default_factory=lambda: deque(maxlen=30))

    def add_episode(self, distance: int, total_reward: float, crashed: bool, epsilon: float):
        self.distances.append(distance)
        self.rewards.append(total_reward)
        self.crashes.append(1 if crashed else 0)
        self.epsilons.append(epsilon)
        self._recent_crashes.append(1 if crashed else 0)

    def moving_avg(self, series: list[float], w: int = 20):
        if len(series) < 2:
            return []
        w = max(2, min(w, len(series)))
        out = []
        s = 0.0
        for i, v in enumerate(series):
            s += v
            if i >= w:
                s -= series[i-w]
            if i >= w-1:
                out.append(s / w)
        return out

    def crash_rate_recent(self):
        if not self._recent_crashes:
            return 0.0
        return sum(self._recent_crashes) / len(self._recent_crashes)
