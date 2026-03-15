from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional

@dataclass
class LiveStats:
    """
    Performance Monitoring and Telemetry system for the RL Agent.
    
    Tracks mission-critical metrics such as survival distance, reward accumulation,
    and operational risk factors across training episodes.

    Attributes:
        distances (List[int]): Number of steps survived per episode.
        rewards (List[float]): Cumulative reward earned per episode.
        crashes (List[int]): Binary indicator of collision (1) or survival (0).
        epsilons (List[float]): Historical exploration rate (epsilon) values.
        risk_history (List[float]): Mean collision risk encountered per episode.
        ttc_history (List[float]): Mean Time-To-Collision (TTC) per episode.
        lane_usage (List[int]): Cumulative step counts per lane [Left, Middle, Right].
        window (int): Size of the rolling average window for trend analysis.
    """
    distances: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    crashes: List[int] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    
    risk_history: List[float] = field(default_factory=list)
    ttc_history: List[float] = field(default_factory=list)
    lane_usage: List[int] = field(default_factory=list)
    
    window: int = 30
    _recent_crashes: deque = field(default_factory=lambda: deque(maxlen=30))

    def __post_init__(self):
        """Initializes lane usage tracking if not provided."""
        if not self.lane_usage:
            self.lane_usage = [0, 0, 0]

    def add_episode(self, distance: int, total_reward: float, crashed: bool, epsilon: float, 
                    avg_risk: float = 0.0, avg_ttc: float = 0.0, final_lane_hist: Optional[List[int]] = None):
        """
        Registers telemetry data from a completed episode.

        Args:
            distance (int): Survival duration in steps.
            total_reward (float): Cumulative reward signal.
            crashed (bool): Terminal state cause (collision vs max steps).
            epsilon (float): Exploration rate used during the episode.
            avg_risk (float): Mean risk coefficient.
            avg_ttc (float): Mean temporal safety margin.
            final_lane_hist (Optional[List[int]]): Sequence of lanes occupied during the episode.
        """
        self.distances.append(distance)
        self.rewards.append(total_reward)
        self.crashes.append(1 if crashed else 0)
        self.epsilons.append(epsilon)
        self.risk_history.append(avg_risk)
        self.ttc_history.append(avg_ttc)
        self._recent_crashes.append(1 if crashed else 0)
        
        if final_lane_hist:
            for l in final_lane_hist:
                if 0 <= l <= 2:
                    self.lane_usage[l] += 1

    def moving_avg(self, series: List[float], w: Optional[int] = None) -> List[float]:
        """
        Computes the simple moving average for a data series.

        Args:
            series (List[float]): Input telemetry data.
            w (int, optional): Window size. Defaults to the class instance window.

        Returns:
            List[float]: The smoothed data series.
        """
        if not w: w = self.window
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

    def crash_rate_recent(self) -> float:
        """
        Calculates the weighted failure rate over the most recent observation window.

        Returns:
            float: Failure probability [0.0 - 1.0].
        """
        if not self._recent_crashes:
            return 0.0
        return sum(self._recent_crashes) / len(self._recent_crashes)

    def generate_insights(self) -> List[str]:
        """
        Analyzes historical data to generate high-level architectural insights.
        
        Uses heuristic evaluation of performance trends, behavioral biases, 
        and risk exposure.

        Returns:
            List[str]: Human-readable technical insights.
        """
        insights = []
        if len(self.distances) < 5:
            return ["Awaiting sufficient telemetry for behavioral analysis..."]

        # KPI 1: Policy Convergence Trend
        recent_dist = sum(self.distances[-5:]) / 5
        early_dist = sum(self.distances[:5]) / 5
        if recent_dist > early_dist * 1.5:
            insights.append("✓ Positive convergence: Significant improvement in evasion policy detected.")
        elif recent_dist < early_dist * 0.8:
            insights.append("⚠ Performance Degresssion: Potential overfitting or suboptimal exploitation phase.")

        # KPI 2: Behavioral Bias Analysis
        total_steps = sum(self.lane_usage)
        if total_steps > 0:
            lanes_pct = [c/total_steps for c in self.lane_usage]
            pref_idx = lanes_pct.index(max(lanes_pct))
            lane_names = ['Left', 'Center', 'Right']
            insights.append(f"ℹ Lateral Bias: Agent shows a {lanes_pct[pref_idx]:.1%} preference for the {lane_names[pref_idx]} lane.")

        # KPI 3: Operational Risk Assessment
        if self.risk_history:
            avg_risk = sum(self.risk_history[-10:]) / 10
            if avg_risk > 0.4:
                insights.append("☢ Risk Alert: Average risk density exceeds nominal thresholds. Review collision penalties.")
            else:
                insights.append("Operational risk metrics remain within nominal efficiency parameters.")

        return insights
