"""Gymnasium interface for the 2D combat arena."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.arena import ACTION_NAMES, AGENT_COOLDOWN, AGENT_MAX_AMMO, ArenaSimulation

OBS_FEATURE_NAMES = [
    "agent_x", "agent_y", "opp_rel_x", "opp_rel_y", "opp_distance",
    "agent_health", "opp_health", "agent_ammo", "shoot_cooldown",
    "has_line_of_sight", "cover_rel_x", "cover_rel_y", "in_cover",
    "threat_rel_x", "threat_rel_y", "threat_present",
]

STEP_PENALTY = -0.005
DAMAGE_DEALT_COEF = 0.025
DAMAGE_TAKEN_COEF = -0.02
INVALID_SHOT_PENALTY = -0.01
BLIND_SHOT_PENALTY = -0.02
DEFENSE_BONUS = 0.02
DEFENSE_BUDGET = 1.0
LOW_HEALTH = 40.0
WIN_REWARD = 5.0
LOSS_REWARD = -5.0


class CombatArenaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, max_steps=400):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.render_mode = render_mode
        self.sim = ArenaSimulation(width=600, height=600)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        low = np.array([0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1, 0], dtype=np.float32)
        high = np.ones(len(OBS_FEATURE_NAMES), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.window = None
        self.clock = None
        self.max_steps = max_steps
        self.current_step = 0
        self._defense_paid = 0.0
        self._last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(rng=self.np_random)
        self.current_step = 0
        self._defense_paid = 0.0
        self._last_action = None
        return self._get_obs(), self._get_info()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected a value in {self.action_space}")
        action = int(action)
        self._last_action = action
        self.current_step += 1
        events = self.sim.step(action)
        components = self._reward_components(events)

        agent_dead = self.sim.agent_health <= 0
        opponent_dead = self.sim.opp_health <= 0
        terminated = agent_dead or opponent_dead
        winner = None
        if terminated:
            if opponent_dead and not agent_dead:
                components["terminal"], winner = WIN_REWARD, "agent"
            elif agent_dead and not opponent_dead:
                components["terminal"], winner = LOSS_REWARD, "opponent"
            else:
                winner = "draw"
        truncated = not terminated and self.current_step >= self.max_steps
        info = self._get_info()
        info.update({"reward_components": components, "winner": winner, "events": events})

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(sum(components.values())), terminated, truncated, info

    def _reward_components(self, events):
        components = {
            "step": STEP_PENALTY,
            "damage_dealt": DAMAGE_DEALT_COEF * events["damage_dealt"],
            "damage_taken": DAMAGE_TAKEN_COEF * events["damage_taken"],
            "invalid_shot": INVALID_SHOT_PENALTY if events["invalid_shot"] else 0.0,
            "blind_shot": BLIND_SHOT_PENALTY if events["fired_blind"] else 0.0,
            "defense": 0.0,
            "terminal": 0.0,
        }
        if (self.sim.agent_health <= LOW_HEALTH and not self.sim.los
                and self._defense_paid < DEFENSE_BUDGET):
            bonus = min(DEFENSE_BONUS, DEFENSE_BUDGET - self._defense_paid)
            components["defense"] = bonus
            self._defense_paid += bonus
        return components

    def _get_obs(self):
        sim = self.sim
        width, height = sim.width, sim.height
        rel = (sim.opp_pos - sim.agent_pos) / np.array([width, height], dtype=np.float32)
        distance = sim.distance() / np.hypot(width, height)
        cover_rel = (sim.cover_point - sim.agent_pos) / np.array([width, height], dtype=np.float32)
        threat = sim.nearest_threat()
        if threat is None:
            threat_rel, threat_present = np.zeros(2, dtype=np.float32), 0.0
        else:
            threat_rel = threat / np.array([width, height], dtype=np.float32)
            threat_present = 1.0
        obs = np.array([
            sim.agent_pos[0] / width, sim.agent_pos[1] / height,
            rel[0], rel[1], distance,
            sim.agent_health / 100.0, sim.opp_health / 100.0,
            sim.agent_ammo / AGENT_MAX_AMMO, sim.agent_cooldown / AGENT_COOLDOWN,
            float(sim.los), cover_rel[0], cover_rel[1], float(sim.in_cover),
            threat_rel[0], threat_rel[1], threat_present,
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self):
        sim = self.sim
        return {
            "step": self.current_step,
            "agent_health": float(sim.agent_health),
            "opp_health": float(sim.opp_health),
            "agent_ammo": int(sim.agent_ammo),
            "agent_pos": [float(v) for v in sim.agent_pos],
            "opp_pos": [float(v) for v in sim.opp_pos],
            "cover_point": [float(v) for v in sim.cover_point],
            "distance_to_opp": sim.distance(),
            "has_line_of_sight": bool(sim.los),
            "in_cover": bool(sim.in_cover),
            "is_low_health": bool(sim.agent_health <= 30.0),
            "incoming_projectiles": sum(1 for p in sim.projectiles if p.owner_id == 2),
            "last_action": None if self._last_action is None else ACTION_NAMES[self._last_action],
        }

    def render(self):
        if self.render_mode is None:
            return None
        import pygame

        sim = self.sim
        canvas = pygame.Surface((sim.width, sim.height))
        canvas.fill((230, 230, 235))
        for obstacle in sim.obstacles:
            obstacle.draw(canvas)
        pygame.draw.line(canvas, (80, 200, 100) if sim.los else (220, 90, 90),
                         tuple(map(int, sim.agent_pos)), tuple(map(int, sim.opp_pos)), 1)
        pygame.draw.circle(canvas, (0, 190, 190), tuple(map(int, sim.cover_point)), 5, 2)
        for projectile in sim.projectiles:
            projectile.draw(canvas)
        pygame.draw.circle(canvas, (40, 120, 240), tuple(map(int, sim.agent_pos)), 14)
        pygame.draw.circle(canvas, (240, 60, 60), tuple(map(int, sim.opp_pos)), 14)
        for position, health, color in ((sim.agent_pos, sim.agent_health, (40, 120, 240)),
                                        (sim.opp_pos, sim.opp_health, (240, 60, 60))):
            x, y = int(position[0]) - 15, int(position[1]) - 26
            pygame.draw.rect(canvas, (60, 60, 60), (x, y, 30, 5))
            pygame.draw.rect(canvas, color, (x, y, int(30 * max(health, 0) / 100), 5))
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((sim.width, sim.height))
                pygame.display.set_caption("XAI Combat Arena")
                self.clock = pygame.time.Clock()
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
