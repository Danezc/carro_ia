import pygame
import sys
import random
import math
from typing import List, Dict, Tuple, Any, Optional
from src.train import Trainer

class GameUI:
    """
    Enhanced AI Performance Dashboard for Car Reinforcement Learning.
    
    Provides a high-fidelity visual interface for monitoring the RL agent's 
    learning progress. Features modern HUD aesthetics, real-time telemetry 
    charts, proximity radar, and automated behavioral insights.

    Attributes:
        trainer (Trainer): The training engine providing state and model data.
        screen (pygame.Surface): Main display surface.
        clock (pygame.time.Clock): Frame rate regulator.
        colors (Dict[str, Tuple]): Professional UI color palette.
    """
    def __init__(self, trainer: Trainer):
        """
        Initializes the UI system and pygame resources.

        Args:
            trainer (Trainer): Pre-configured training instance.
        """
        self.trainer = trainer
        pygame.init()
        self.width = 1100
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Neural-X Performance Dashboard | AI Research Lab")
        self.clock = pygame.time.Clock()
        
        # Typography configuration
        self.font_main = pygame.font.SysFont("Segoe UI", 16)
        self.font_bold = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font_title = pygame.font.SysFont("Segoe UI", 24, bold=True)
        self.font_mono = pygame.font.SysFont("Consolas", 14)

        # Environment layout specs
        self.lane_width = 80
        self.padding_left = 60
        self.road_height = 550
        
        # Design System: Cyber-Noir Aesthetics
        self.colors = {
            "bg": (14, 14, 18),
            "panel": (25, 25, 35, 180), # Translucent panel
            "border": (60, 60, 80),
            "accent": (0, 255, 180),   # Neon Cyan
            "danger": (255, 60, 100),  # Alert Red
            "warning": (255, 180, 0),  # Warning Gold
            "text": (230, 230, 240),
            "text_dim": (140, 140, 160),
            "road": (20, 20, 26),
            "car": (0, 255, 180),
        }

        # Dynamic background elements
        self.stars = [
            {"x": random.randint(0, self.width), "y": random.randint(0, self.height), "s": random.uniform(1, 3)}
            for _ in range(100)
        ]

    def _draw_glass_panel(self, x: int, y: int, w: int, h: int, title: str = ""):
        """
        Renders a semi-transparent 'glassmorphism' effect container.

        Args:
            x (int): Horizontal position.
            y (int): Vertical position.
            w (int): Width.
            h (int): Height.
            title (str): Panel header text.
        """
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(s, self.colors["panel"], (0, 0, w, h), border_radius=12)
        pygame.draw.rect(s, self.colors["border"], (0, 0, w, h), 1, border_radius=12)
        self.screen.blit(s, (x, y))
        
        if title:
            txt = self.font_bold.render(title, True, self.colors["accent"])
            self.screen.blit(txt, (x + 15, y + 10))

    def _draw_line_chart(self, x: int, y: int, w: int, h: int, values: List[float], color: Tuple, label: str = ""):
        """
        Visualizes a dynamic performance trend using a line graph.

        Args:
            x (int): Graph x-coordinate.
            y (int): Graph y-coordinate.
            w (int): Graph width.
            h (int): Graph height.
            values (List[float]): Historical metric data.
            color (Tuple): Line color.
            label (str): Metric identifier.
        """
        if len(values) < 2:
            return
        
        pygame.draw.rect(self.screen, (30, 30, 40), (x, y, w, h), border_radius=8)
        
        limit_data = values[-100:]
        vmin, vmax = min(limit_data), max(limit_data)
        if vmax == vmin: vmax += 1e-5
        
        pts = []
        for i, v in enumerate(limit_data):
            px = x + (i * (w - 10) / (len(limit_data) - 1)) + 5
            py = y + h - ((v - vmin) * (h - 15) / (vmax - vmin)) - 5
            pts.append((float(px), float(py)))
        
        if len(pts) > 1:
            pygame.draw.lines(self.screen, color, False, pts, 2)
            
        txt = self.font_mono.render(label, True, (200, 200, 200))
        self.screen.blit(txt, (x + 5, y - 18))

    def _draw_radar(self, x: int, y: int, size: int):
        """
        Renders a proximity sonar visualization based on environmental sensors.

        Args:
            x (int): Radar x-coordinate.
            y (int): Radar y-coordinate.
            size (int): Diameter of the radar interface.
        """
        center = (x + size//2, y + size//2)
        pygame.draw.circle(self.screen, (40, 40, 50), center, size//2, 1)
        pygame.draw.circle(self.screen, (30, 30, 40), center, size//4, 1)
        
        dists = self.trainer.env.get_raw_distances()
        horizon = self.trainer.env.horizon
        
        # Field of View angles for the 3 lanes
        angles = [-30, 0, 30]
        for i, d in enumerate(dists):
            norm = 1.0 - (d / horizon)
            length = (size//2) * norm
            
            rad = math.radians(angles[i] - 90)
            end_x = center[0] + math.cos(rad) * length
            end_y = center[1] + math.sin(rad) * length
            
            col = self.colors["accent"] if norm < 0.7 else self.colors["danger"]
            pygame.draw.line(self.screen, col, center, (end_x, end_y), 3)
            pygame.draw.circle(self.screen, col, (int(end_x), int(end_y)), 4)

    def _draw_metrics(self, x: int, y: int):
        """Displays key-value KPIs in the primary telemetry panel."""
        s = self.trainer.stats
        ep = len(s.distances)
        crash_rate = s.crash_rate_recent() * 100
        
        lines = [
            ("Episodes Trained", f"{ep}"),
            ("Recent Success Rate", f"{100-crash_rate:.1f}%"),
            ("Agent Epsilon", f"{self.trainer.agent.epsilon:.3f}"),
            ("Exploration Phase", "DECAYING" if self.trainer.agent.epsilon > 0.1 else "EXPLOIT")
        ]
        
        curr_y = y + 40
        for label, val in lines:
            self.screen.blit(self.font_main.render(label, True, self.colors["text_dim"]), (x+15, curr_y))
            v_txt = self.font_mono.render(val, True, self.colors["text"])
            self.screen.blit(v_txt, (x + 230 - v_txt.get_width(), curr_y))
            curr_y += 24

    def _draw_road(self):
        """Renders the highway environment structural elements."""
        mx, my = self.padding_left, 50
        pygame.draw.rect(self.screen, self.colors["road"], (mx, my, 3 * self.lane_width, self.road_height))
        pygame.draw.rect(self.screen, self.colors["border"], (mx-2, my, 3 * self.lane_width+4, self.road_height), 2)
        
        for i in range(1, 3):
            lx = mx + i * self.lane_width
            pygame.draw.line(self.screen, (50, 50, 70), (lx, my), (lx, my + self.road_height), 1)

    def _draw_car(self):
        """Visualizes the agent's vehicle and its dynamic collision risk aura."""
        env = self.trainer.env
        lane = env.car_lane
        w, h = 34, 45
        center_x = self.padding_left + lane * self.lane_width + (self.lane_width // 2)
        x = center_x - (w // 2)
        y = 50 + self.road_height - 70
        
        pygame.draw.ellipse(self.screen, (0, 0, 0, 100), (x, y+h-5, w, 15))
        pygame.draw.rect(self.screen, self.colors["car"], (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, (20, 40, 40), (x+4, y+6, w-8, 14), border_radius=4)
        
        # Risk Aura: Visual indication of immediate danger level
        risk = env.get_collision_risk()
        if risk > 0.3:
            alpha = int(min(255, risk * 400))
            aura = pygame.Surface((w+20, h+20), pygame.SRCALPHA)
            pygame.draw.rect(aura, (255, 60, 100, alpha//2), (0, 0, w+20, h+20), border_radius=12)
            self.screen.blit(aura, (x-10, y-10))

    def _draw_obstacles(self):
        """Renders upcoming traffic obstacles based on environment coordinates."""
        step_size = self.road_height / self.trainer.env.horizon
        car_draw_y = 50 + self.road_height - 70
        
        for ob in self.trainer.env.obstacles:
            center_x = self.padding_left + ob.lane * self.lane_width + (self.lane_width // 2)
            visual_y = car_draw_y - (ob.y * step_size)
            
            if 0 < visual_y < self.height:
                color = [(220, 70, 70), (220, 150, 50), (180, 50, 200)][ob.color_idx % 3]
                pygame.draw.circle(self.screen, color, (center_x, int(visual_y)), 18)
                pygame.draw.circle(self.screen, (255, 255, 255, 100), (center_x-5, int(visual_y)-5), 4)

    def run(self):
        """
        The main application loop.
        
        Synchronizes input handling, physics updates (when auto-play is on), 
        and dashboard rendering at 30 FPS.
        """
        running = True
        auto_play = False
        crash_timer = 0
        current_ep_reward = 0.0
        
        while running:
            # 1. Background Animation
            self.screen.fill(self.colors["bg"])
            for s in self.stars:
                s["y"] = (s["y"] + s["s"]) % self.height
                pygame.draw.circle(self.screen, (60, 60, 80), (int(s["x"]), int(s["y"])), 1)

            # 2. Event Dispatcher
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t: self.trainer.train(n_episodes=50)
                    if event.key == pygame.K_p: auto_play = not auto_play
                    if event.key == pygame.K_r: 
                        self.trainer.env.reset()
                        current_ep_reward = 0.0

            # 3. Agent & Environment Synchronization
            if auto_play and crash_timer == 0:
                env = self.trainer.env
                if env.done:
                    env.reset()
                    current_ep_reward = 0.0
                
                state = env.state()
                action = self.trainer.agent.act(state, training=False)
                s2, r, done, info = env.step(action)
                current_ep_reward += r
                
                if done:
                    # Update live telemetry metrics
                    self.trainer.stats.add_episode(
                        distance=info["distance"],
                        total_reward=current_ep_reward,
                        crashed=info["crashed"],
                        epsilon=self.trainer.agent.epsilon,
                        avg_risk=info["risk"],
                        avg_ttc=info["min_ttc"],
                        final_lane_hist=env.lane_history
                    )
                    if info["crashed"]: crash_timer = 20

            # 4. Global Rendering Pipeline
            self._draw_road()
            self._draw_obstacles()
            self._draw_car()
            
            if crash_timer > 0:
                crash_timer -= 1
                overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                overlay.fill((255, 0, 0, crash_timer * 5))
                self.screen.blit(overlay, (0,0))

            # HUD Components
            panel_x = 350
            self._draw_glass_panel(panel_x, 50, 300, 240, title="Core Telemetry")
            self._draw_metrics(panel_x, 50)
            self._draw_radar(panel_x + 180, 150, 100)

            self._draw_glass_panel(panel_x, 310, 300, 290, title="Neural Progression")
            st = self.trainer.stats
            self._draw_line_chart(panel_x+15, 360, 270, 70, [float(d) for d in st.distances], self.colors["accent"], "Performance (Dist)")
            self._draw_line_chart(panel_x+15, 455, 270, 70, st.moving_avg([float(r) for r in st.rewards], 20), self.colors["warning"], "Reward Trend (avg)")
            self._draw_line_chart(panel_x+15, 550, 270, 30, [float(ri) for ri in st.risk_history], self.colors["danger"], "Risk Density")

            ins_x = 670
            self._draw_glass_panel(ins_x, 50, 400, 550, title="AI Analyst Insights")
            insights = st.generate_insights()
            for i, line in enumerate(insights):
                color = self.colors["text"]
                if "⚠" in line: color = self.colors["warning"]
                if "☢" in line: color = self.colors["danger"]
                if "✓" in line: color = self.colors["accent"]
                
                txt_surf = self.font_main.render(line, True, color)
                self.screen.blit(txt_surf, (ins_x + 15, 100 + i*35))

            ctrl_txt = "SPACE: Toggle Manual | P: Toggle Autoplay | T: Fast Train (50) | R: Reset"
            self.screen.blit(self.font_mono.render(ctrl_txt, True, self.colors["text_dim"]), (self.padding_left, 620))
            
            status = "ANALYZING..." if auto_play else "STANDBY"
            stat_col = self.colors["accent"] if auto_play else self.colors["warning"]
            pygame.draw.rect(self.screen, stat_col, (self.padding_left, 10, 10, 30))
            self.screen.blit(self.font_bold.render(status, True, stat_col), (self.padding_left + 20, 15))

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()
