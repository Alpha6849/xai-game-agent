import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from env.arena import ArenaSimulation

class CombatArenaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.sim = ArenaSimulation(width=600, height=600)
        self.render_mode = render_mode

        # 7 Discrete Actions: [North, South, East, West, Shoot, Cover, Retreat]
        self.action_space = spaces.Discrete(7)

        # 7-dim Normalized Observation Space
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.window = None
        self.clock = None
        self.max_steps = 400
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        rel_x = (self.sim.opp_pos[0] - self.sim.agent_pos[0]) / self.sim.width
        rel_y = (self.sim.opp_pos[1] - self.sim.agent_pos[1]) / self.sim.height
        cover_center = np.array([300.0, 300.0])
        dist_cover = np.linalg.norm(self.sim.agent_pos - cover_center) / np.hypot(self.sim.width, self.sim.height)

        return np.array([
            self.sim.agent_pos[0] / self.sim.width,
            self.sim.agent_pos[1] / self.sim.height,
            rel_x,
            rel_y,
            self.sim.agent_health / 100.0,
            self.sim.agent_ammo / 30.0,
            dist_cover
        ], dtype=np.float32)

    def _get_info(self):
        return {
            "agent_health": self.sim.agent_health,
            "opp_health": self.sim.opp_health,
            "agent_ammo": self.sim.agent_ammo,
            "is_low_health": bool(self.sim.agent_health <= 30.0),
            "step": self.current_step
        }

    def step(self, action):
        self.current_step += 1
        reward = -0.01  # Stalling penalty

        self.sim.perform_action(action)
        hit = self.sim.update_projectiles()

        if hit:
            reward += 0.5

        if action == 5 and self.sim.agent_health <= 30.0:
            reward += 0.3

        terminated = False
        if self.sim.opp_health <= 0:
            reward += 10.0
            terminated = True
        elif self.sim.agent_health <= 0:
            reward -= 10.0
            terminated = True

        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.sim.width, self.sim.height))
            pygame.display.set_caption("XAI Combat Arena")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.sim.width, self.sim.height))
        canvas.fill((230, 230, 235))

        for obs in self.sim.obstacles:
            obs.draw(canvas)
        for proj in self.sim.projectiles:
            proj.draw(canvas)

        pygame.draw.circle(canvas, (40, 120, 240), self.sim.agent_pos.astype(int), 14)
        pygame.draw.circle(canvas, (240, 60, 60), self.sim.opp_pos.astype(int), 14)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()