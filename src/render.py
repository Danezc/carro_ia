import pygame
import sys
from src.train import Trainer

class GameUI:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        pygame.init()
        self.width = 900
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AI Car Learning - Live Graphs")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        # Config visual
        self.lane_width = 80
        self.padding_left = 50
        self.road_height = 500
        
        # Colors
        self.bg_color = (20, 20, 24)
        self.road_color = (40, 40, 45)
        self.car_color = (100, 200, 100)
        self.obs_color = (200, 80, 80)
    
    def _draw_line_chart(self, screen, x, y, w, h, values, color=(240,240,240), label=""):
        pygame.draw.rect(screen, (28, 28, 34), (x, y, w, h), border_radius=8)
        pygame.draw.rect(screen, (80, 80, 90), (x, y, w, h), 1, border_radius=8)

        if len(values) < 2:
            return

        vmin = min(values)
        vmax = max(values)
        if vmax == vmin:
            vmax = vmin + 1e-6

        # map a coords
        pts = []
        n = len(values)
        # Use last w points or all if len < w (using w as logical window approx pixels width too?)
        # User snippet said "values[-w:]", assuming user passes a slice or we slice here.
        # But if W is width in pixels, maybe we want to fit more?
        # Let's trust user logic: "for i, v in enumerate(values[-w:]):" where w was width passed to arg?
        # No, in user snippet w was width of rect but slicing used `w` as count too?
        # User code: "for i, v in enumerate(values[-w:]): ... int(i * (w-10)..."
        # Wait, if `w` argument is pixel width (190), then `values[-w:]` takes last 190 points.
        # That seems reasonable.
        
        limit_data = values[-int(w):] # Take at most `w` points to match pixels roughly
        
        for i, v in enumerate(limit_data): 
            px = x + int(i * (w-10) / max(1, len(limit_data)-1)) + 5
            py = y + h - int((v - vmin) * (h-10) / (vmax - vmin)) - 5
            pts.append((px, py))

        for i in range(1, len(pts)):
            pygame.draw.line(screen, color, pts[i-1], pts[i], 2)

        if label:
            small_font = pygame.font.SysFont("consolas", 14)
            screen.blit(small_font.render(label, True, (220,220,220)), (x+8, y+6))
            screen.blit(small_font.render(f"min={vmin:.1f} max={vmax:.1f}", True, (160,160,160)), (x+8, y+26))

    def _draw_metrics(self, screen, x, y):
        font = pygame.font.SysFont("consolas", 16)
        big = pygame.font.SysFont("consolas", 20, bold=True)

        s = self.trainer.stats
        ep = len(s.distances)
        crash_rate = s.crash_rate_recent() * 100

        last_dist = s.distances[-1] if ep else 0
        avg20 = s.moving_avg([float(d) for d in s.distances], w=20)
        last_avg20 = avg20[-1] if avg20 else 0

        screen.blit(big.render("Live Stats", True, (255,255,255)), (x, y))
        y += 30
        screen.blit(font.render(f"Episodios: {ep}", True, (230,230,230)), (x, y)); y += 22
        screen.blit(font.render(f"Última distancia: {last_dist}", True, (230,230,230)), (x, y)); y += 22
        screen.blit(font.render(f"Media móvil (20): {last_avg20:.1f}", True, (230,230,230)), (x, y)); y += 22
        screen.blit(font.render(f"Crash rate (últ {s.window}): {crash_rate:.1f}%", True, (255,200,200)), (x, y)); y += 22
        screen.blit(font.render(f"Epsilon: {self.trainer.agent.epsilon:.3f}", True, (200,200,255)), (x, y)); y += 22

    def _draw_car(self, screen, lane):
        # Smaller car to ensure clear gaps
        w = 40 # Was 60
        h = 36 # Was 40
        # Center in lane
        center_x = self.padding_left + lane * self.lane_width + (self.lane_width // 2)
        x = center_x - (w // 2)
        # Position at bottom
        y = 50 + self.road_height - 50
        
        pygame.draw.rect(screen, self.car_color, (x, y, w, h), border_radius=6)
        # windshield
        pygame.draw.rect(screen, (40, 60, 80), (x+5, y+5, w-10, 12), border_radius=3)

    def _draw_obstacles(self, screen, obstacles):
        # Logic: y=0 is AT CAR. y=1 is one step away.
        # Car Top is roughly at: 50 + road_height - 50 = Bottom - 50.
        car_top = 50 + self.road_height - 50
        step_size = 50 
        
        for ob in obstacles:
            lane = ob.lane
            
            # visual_y is Top of obstacle? No, let's say Center.
            # Let's keep rect logic for position but draw circle.
            
            visual_y = car_top - (ob.y * step_size)
            
            # Draw as Circle! easier to see "misses"
            center_x = self.padding_left + lane * self.lane_width + (self.lane_width // 2)
            radius = 16 # Diameter 32 (Lane is 80) -> HUGE GAP
            
            # Visual Y is the bottom edge? Or top?
            # Let's say visual_y is the BOTTOM of the shape (closest to car)
            # So CenterY = visual_y - radius
            
            cy = visual_y - radius
            
            # Draw only if visible (roughly)
            if cy + radius > 50:
                 pygame.draw.circle(screen, self.obs_color, (center_x, cy), radius)
                 pygame.draw.circle(screen, (150, 50, 50), (center_x, cy), radius-4)

    def run(self):
        running = True
        auto_play = False  # If True, play episodes visibly
        
        # For crash effect
        crash_timer = 0
        
        # Accumulators for auto-play stats
        current_ep_reward = 0.0
        
        while running:
            self.screen.fill(self.bg_color)
            
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        # Train batch
                        self.trainer.train(n_episodes=50) # Train 50 eps on keypress
                    elif event.key == pygame.K_p:
                        # Toggle auto play
                        auto_play = not auto_play
                        if auto_play:
                            # Reset if needed or just continue
                            if self.trainer.env.done:
                                self.trainer.env.reset()
                                current_ep_reward = 0.0
                    elif event.key == pygame.K_r:
                         # Reset env manually
                         self.trainer.env.reset()
                         current_ep_reward = 0.0

            # Logic for Auto-Play (Visual Demo)
            crashed_this_frame = False
            if auto_play and crash_timer == 0:
                # One step per frame
                env = self.trainer.env
                if env.done:
                    # Save Stats for the finished episode!
                    # Logic assumes we just finished in previous frame? 
                    # Actually logic: step -> done=True. Next frame we enter here.
                    # We should record it NOW before reset.
                    # But wait, we need 'info' from the last step.
                    # Easier: record right after step if done.
                    
                    # Reset
                    env.reset()
                    current_ep_reward = 0.0
                
                state = env.state()
                action = self.trainer.agent.act(state, training=False) 
                
                s2, r, done, info = env.step(action)
                current_ep_reward += r
                
                if done:
                    # Record Stat
                    self.trainer.stats.add_episode(
                        distance=info["distance"],
                        total_reward=current_ep_reward,
                        crashed=info["crashed"],
                        epsilon=self.trainer.agent.epsilon
                    )

                if info["crashed"]:
                    crashed_this_frame = True
                    crash_timer = 15 # Show crash for 15 frames

            # Draw Road
            mx = self.padding_left
            my = 50
            pygame.draw.rect(self.screen, self.road_color, (mx, my, 3 * self.lane_width, self.road_height))
            
            # Draw dividers
            pygame.draw.line(self.screen, (100,100,100), (mx + self.lane_width, my), (mx + self.lane_width, my + self.road_height), 2)
            pygame.draw.line(self.screen, (100,100,100), (mx + 2*self.lane_width, my), (mx + 2*self.lane_width, my + self.road_height), 2)
            
            # Draw Objects
            self._draw_obstacles(self.screen, self.trainer.env.obstacles)
            self._draw_car(self.screen, self.trainer.env.car_lane)
            
            # Visual Crash Effect
            if crash_timer > 0:
                crash_timer -= 1
                # Draw explosion logic
                # Get car pos
                cx = self.padding_left + self.trainer.env.car_lane * self.lane_width + self.lane_width // 2
                cy = 50 + self.road_height - 40 # approx car center
                # Big red circle
                pygame.draw.circle(self.screen, (255, 50, 50), (cx, cy), 40 + (15-crash_timer)*2, 4)
                pygame.draw.circle(self.screen, (255, 100, 0), (cx, cy), 20 + (15-crash_timer), 0)
                
                f = pygame.font.SysFont("consolas", 40, bold=True)
                txt = f.render("CRASH!", True, (255, 255, 0))
                self.screen.blit(txt, (cx - 60, cy - 80))

            # Draw Info Text
            status = "PLAYING (Greedy)" if auto_play else "PAUSED"
            txt = self.font.render(f"Press 'T' to train 50 episodes fast. 'P' to toggle Play/Pause. Status: {status}", True, (255,255,255))
            self.screen.blit(txt, (mx, my + self.road_height + 20))

            # Draw Current Distance on Top of Road
            dist_txt = self.font.render(f"Distance: {self.trainer.env.step_count}", True, (255, 255, 255))
            self.screen.blit(dist_txt, (mx + 10, my + 10))
            
            # === Gráficas en vivo (si hay stats) ===
            st = self.trainer.stats
            
            # Panel derecho para charts
            chart_x = 400
            chart_y = 50

            self._draw_metrics(self.screen, chart_x, chart_y)
            chart_y += 140

            # Distancia por episodio
            self._draw_line_chart(
                self.screen, chart_x, chart_y, 300, 100,
                [float(d) for d in st.distances],
                color=(180, 255, 180),
                label="Distancia/episodio"
            )
            chart_y += 110

            # Media móvil distancia
            ma = st.moving_avg([float(d) for d in st.distances], w=20)
            self._draw_line_chart(
                self.screen, chart_x, chart_y, 300, 100,
                ma,
                color=(255, 220, 140),
                label="Media móvil (20)"
            )
            chart_y += 110

            # Epsilon
            self._draw_line_chart(
                self.screen, chart_x, chart_y, 300, 100,
                [float(e) for e in st.epsilons],
                color=(180, 200, 255),
                label="Epsilon"
            )

            pygame.display.flip()
            self.clock.tick(15 if auto_play else 60) # Slow down if playing to see

        pygame.quit()
        sys.exit()
