import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Obstacle:
    lane: int
    y: int

class LaneEnv:
    def __init__(self, horizon: int = 12, spawn_prob: float = 0.35, seed: int = 7):
        self.horizon = horizon
        self.spawn_prob = spawn_prob
        self.rng = random.Random(seed)
        
        # State
        self.car_lane = 1  # 0, 1, 2
        self.obstacles: List[Obstacle] = []
        self.step_count = 0
        self.done = False

    def reset(self):
        self.car_lane = 1
        self.obstacles = []
        self.step_count = 0
        self.done = False
        return self.state()

    def _bin_dist(self, d: int) -> int:
        # 0..5 based on distance
        # 0: very close/crash, 1: close, ... 5: far
        if d <= 1: return 0
        mask = [2, 4, 6, 8, 10]
        # if d <= 2 return 1, if d <= 4 return 2...
        for i, val in enumerate(mask):
            if d <= val:
                return i + 1
        return 5  # > 10

    def state(self):
        """
        Estado mejorado para aprender rápido:
        - car_lane: 0..2
        - nearest_dist_left_bin: 0..5
        - nearest_dist_mid_bin: 0..5
        - nearest_dist_right_bin: 0..5
        """
        dists = [self.horizon, self.horizon, self.horizon]
        for ob in self.obstacles:
            # Distance from car (y=0 implicito relative to bottom if we think logic, 
            # but usually in these games y goes 0 to horizon. Let's assume car is at y=0 and obstacles at y>0)
            # Or if visual is top-down: car at bottom.
            # Let's simplify: distance = ob.y - car_y.
            # Actually user snippet says "ob.y <= dists[ob.lane]", implying ob.y is the distance? 
            # Let's assume obstacles come from horizon down to 0, car is at 0.
            # Wait, if ob.y is distance, then 0 is crash.
            
            if 0 <= ob.y <= dists[ob.lane]:
                dists[ob.lane] = ob.y

        return (self.car_lane, self._bin_dist(dists[0]), self._bin_dist(dists[1]), self._bin_dist(dists[2]))

    def step(self, action: int):
        # Action: 0=Left, 1=Stay, 2=Right
        if action == 0:
            self.car_lane = max(0, self.car_lane - 1)
        elif action == 2:
            self.car_lane = min(2, self.car_lane + 1)
        
        # Move obstacles (decrease y)
        # We spawn at `horizon`, they move to 0. Car is at 0.
        # So "distance" is exactly y.
        
        crashed = False
        new_obs = []
        for ob in self.obstacles:
            ob.y -= 1
            if ob.y < 0:
                continue # Passed
            
            # Check collision
            if ob.y == 0 and ob.lane == self.car_lane:
                crashed = True
            
            new_obs.append(ob)
        
        self.obstacles = new_obs
        
        # Spawn new?
        # Enforce "one at a time" / minimum gap
        min_gap = 3
        too_close = any(ob.y > (self.horizon - min_gap) for ob in self.obstacles)

        if not too_close and self.rng.random() < self.spawn_prob:
            # Pick lane
            l = self.rng.randint(0, 2)
            # Avoid impossible walls logic if wanted, but simpler:
            self.obstacles.append(Obstacle(lane=l, y=self.horizon))
            
        self.step_count += 1
        
        reward = 1.0
        if crashed:
            reward = -10.0 # Punishment
            self.done = True
        
        info = {
            "crashed": crashed,
            "distance": self.step_count
        }
        
        return self.state(), reward, self.done, info
