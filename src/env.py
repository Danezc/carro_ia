import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

@dataclass
class Obstacle:
    """
    Represents an obstacle (traffic car) in the highway.
    
    Attributes:
        lane (int): The lane index [0, 1, 2].
        y (int): The vertical position relative to the horizon. y=0 is the car's collision point.
        color_idx (int): Visual variation index for rendering purposes.
    """
    lane: int
    y: int
    color_idx: int = 0


class LaneEnv:
    """
    A 3-lane highway environment simulation designed for Reinforcement Learning.
    
    The environment simulates a simplified highway where the agent (blue car) must
    evade oncoming obstacles. Obstacles spawn at the horizon and move toward the 
    agent at a constant velocity. The state space is discretized for tabular RL.

    Attributes:
        horizon (int): The maximum vertical distance an obstacle can be from the agent.
        spawn_prob (float): Probability of a new obstacle spawning in each step.
        car_lane (int): Current lane of the agent [0: Left, 1: Middle, 2: Right].
        obstacles (List[Obstacle]): List of active obstacles in the environment.
        step_count (int): Counter for the number of steps in the current episode.
        done (bool): Terminal state flag.
    """

    def __init__(self, horizon: int = 14, spawn_prob: float = 0.32, seed: int = 7):
        """
        Initializes the environment with specific traffic density and horizon.

        Args:
            horizon (int): Vertical grid size. Defaults to 14.
            spawn_prob (float): Traffic density factor. Defaults to 0.32.
            seed (int): Random seed for reproducible traffic patterns. Defaults to 7.
        """
        self.horizon = horizon
        self.spawn_prob = spawn_prob
        self.rng = random.Random(seed)

        self.car_lane: int = 1
        self.obstacles: List[Obstacle] = []
        self.step_count: int = 0
        self.done: bool = False
        self.lane_history: List[int] = []

    def reset(self) -> Tuple[int, int, int, int]:
        """
        Resets the environment to its initial state.

        Returns:
            Tuple[int, int, int, int]: The initial state observation.
        """
        self.car_lane = 1
        self.obstacles = []
        self.step_count = 0
        self.done = False
        self.lane_history = []
        return self.state()

    def _bin_dist(self, d: int) -> int:
        """
        Discretizes a raw distance value into one of 6 discrete bins.
        
        This enables a finite state space for the tabular Q-learning agent.
        0: Imminent collision (1.0 - (d / horizon))
        5: Clear road ahead.

        Args:
            d (int): Raw vertical distance.

        Returns:
            int: The discretized distance bin [0-5].
        """
        if d <= 1:
            return 0
        thresholds = [2, 4, 6, 9, 12]
        for i, t in enumerate(thresholds):
            if d <= t:
                return i + 1
        return 5

    def state(self) -> Tuple[int, int, int, int]:
        """
        Constructs the current state representation.

        The state is a 4-tuple: (car_lane, bin_dist_L, bin_dist_M, bin_dist_R).

        Returns:
            Tuple[int, int, int, int]: The discretized state.
        """
        dists = [self.horizon, self.horizon, self.horizon]
        for ob in self.obstacles:
            if 0 <= ob.y < dists[ob.lane]:
                dists[ob.lane] = ob.y
        return (
            self.car_lane,
            self._bin_dist(dists[0]),
            self._bin_dist(dists[1]),
            self._bin_dist(dists[2]),
        )

    def get_raw_distances(self) -> List[int]:
        """
        Computes the raw vertical distance to the closest obstacle in each lane.

        Returns:
            List[int]: A list of 3 distances [Left, Middle, Right].
        """
        dists = [self.horizon, self.horizon, self.horizon]
        for ob in self.obstacles:
            if 0 <= ob.y < dists[ob.lane]:
                dists[ob.lane] = ob.y
        return dists

    def get_ttc(self) -> List[float]:
        """
        Calculates Time-To-Collision (TTC) for each lane.
        
        Assumes constant unit velocity (1 unit per step). 
        TTC = Distance / Velocity.

        Returns:
            List[float]: TTC values for each lane.
        """
        dists = self.get_raw_distances()
        # Relative speed is 1 step per frame
        ttc = [d / 1.0 for d in dists]
        return ttc

    def get_collision_risk(self) -> float:
        """
        Computes a dynamic collision risk factor based on obstacle proximity.

        The risk is calculated using an exponential decay model: 
        Risk = 1 / (1 + Distance).

        Returns:
            float: A risk coefficient in range [0.0, 1.0].
        """
        dists = self.get_raw_distances()
        current_lane_dist = dists[self.car_lane]
        risk = 1.0 / (1.0 + current_lane_dist)
        return float(risk)

    def step(self, action: int) -> Tuple[Tuple[int, int, int, int], float, bool, Dict[str, Any]]:
        """
        Performs a single simulation step based on the agent's action.

        Args:
            action (int): The requested maneuver (0: Left, 1: Stay, 2: Right).

        Returns:
            Tuple: A 4-tuple containing:
                - next_state: The new environment state.
                - reward: Magnitude of reward/penalty.
                - done: Whether the episode has terminated.
                - info: Metadata dictionary with telemetry and analytics.
        """
        # Execute lateral movement
        if action == 0:
            self.car_lane = max(0, self.car_lane - 1)
        elif action == 2:
            self.car_lane = min(2, self.car_lane + 1)

        # Physics: Advance obstacles and detect collisions
        crashed = False
        survivors = []
        for ob in self.obstacles:
            ob.y -= 1
            if ob.y < 0:
                continue
            if ob.y == 0 and ob.lane == self.car_lane:
                crashed = True
            survivors.append(ob)
        self.obstacles = survivors

        # Traffic Generation: Spawn new obstacles
        min_gap = 3
        too_close = any(ob.y > (self.horizon - min_gap) for ob in self.obstacles)
        if not too_close and self.rng.random() < self.spawn_prob:
            self.obstacles.append(Obstacle(
                lane=self.rng.randint(0, 2),
                y=self.horizon,
                color_idx=self.rng.randint(0, 3),
            ))

        self.step_count += 1
        self.lane_history.append(self.car_lane)

        # Reward Engineering
        if crashed:
            reward = -20.0  # Significant penalty for catastrophic failure
            self.done = True
        else:
            reward = 1.0    # Baseline survival reward
            
            # Behavioral shaping: Encourage lane stability and defensive driving
            if self.car_lane == 1:
                reward += 0.1  # Preference for center lane (safety buffer)
            
            if action != 1:
                reward -= 0.1  # Penalty for erratic lane changing (smoothness)

        # Telemetry aggregation
        ttc = self.get_ttc()
        info = {
            "crashed": crashed,
            "distance": self.step_count,
            "action": action,
            "risk": self.get_collision_risk(),
            "ttc": ttc,
            "min_ttc": min(ttc)
        }
        return self.state(), reward, self.done, info
