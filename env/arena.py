"""2D combat arena simulation, independent of Gymnasium."""
import numpy as np
import pygame

ENTITY_RADIUS = 14
WALL_MARGIN = 20.0
AGENT_SPEED = 5.0
OPP_SPEED = 3.5
PROJ_SPEED = 8.0
HIT_RADIUS = 18.0
MUZZLE_OFFSET = 20.0
AGENT_DAMAGE = 20.0
OPP_DAMAGE = 10.0
AGENT_COOLDOWN = 12
OPP_COOLDOWN = 10
AGENT_MAX_AMMO = 30
AMMO_REGEN_INTERVAL = 15
OPP_AIM_NOISE = 0.08
OPP_MAX_RANGE = 400.0
COVER_GRID_STEP = 30

ACTION_NAMES = ["north", "south", "east", "west", "shoot", "take_cover", "retreat"]


class Obstacle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, surface):
        pygame.draw.rect(surface, (120, 120, 120), self.rect)
        pygame.draw.rect(surface, (70, 70, 70), self.rect, 2)


class Projectile:
    def __init__(self, x, y, dx, dy, speed=PROJ_SPEED, owner_id=1):
        self.x, self.y = float(x), float(y)
        norm = max(float(np.hypot(dx, dy)), 1e-5)
        self.vx, self.vy = (dx / norm) * speed, (dy / norm) * speed
        self.owner_id = owner_id
        self.radius = 4
        self.active = True

    def update(self, width, height, obstacles):
        self.x += self.vx
        self.y += self.vy
        if not (0 <= self.x <= width and 0 <= self.y <= height):
            self.active = False
            return
        rect = pygame.Rect(int(self.x - self.radius), int(self.y - self.radius),
                           self.radius * 2, self.radius * 2)
        if any(o.rect.colliderect(rect) for o in obstacles):
            self.active = False

    def draw(self, surface):
        if self.active:
            color = (255, 200, 50) if self.owner_id == 1 else (255, 90, 40)
            pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.radius)


class ArenaSimulation:
    def __init__(self, width=600, height=600, rng=None):
        self.width, self.height = int(width), int(height)
        if self.width <= 2 * WALL_MARGIN or self.height <= 2 * WALL_MARGIN:
            raise ValueError("Arena dimensions must be greater than twice WALL_MARGIN")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.obstacles = [Obstacle(250, 220, 100, 160), Obstacle(100, 100, 60, 60),
                          Obstacle(440, 440, 60, 60)]
        self._cover_grid = self._build_cover_grid()
        if not len(self._cover_grid):
            raise ValueError("Arena has no free cover-grid positions")
        self.reset()

    def _free(self, pos):
        r = ENTITY_RADIUS
        rect = pygame.Rect(int(pos[0] - r), int(pos[1] - r), 2 * r, 2 * r)
        return not any(o.rect.colliderect(rect) for o in self.obstacles)

    def _clamp(self, pos):
        pos = np.array(pos, dtype=np.float32)
        pos[0] = np.clip(pos[0], WALL_MARGIN, self.width - WALL_MARGIN)
        pos[1] = np.clip(pos[1], WALL_MARGIN, self.height - WALL_MARGIN)
        return pos

    def _move(self, pos, delta):
        delta = np.asarray(delta, dtype=np.float32)
        candidate = self._clamp(pos + delta)
        if self._free(candidate):
            return candidate
        slid = np.array(pos, dtype=np.float32)
        for axis in (0, 1):
            trial = slid.copy()
            trial[axis] += delta[axis]
            trial = self._clamp(trial)
            if self._free(trial):
                slid = trial
        return slid

    def has_line_of_sight(self, a, b):
        p1, p2 = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
        return not any(o.rect.clipline(p1, p2) for o in self.obstacles)

    def _build_cover_grid(self):
        pts = []
        for x in range(int(WALL_MARGIN), int(self.width - WALL_MARGIN) + 1, COVER_GRID_STEP):
            for y in range(int(WALL_MARGIN), int(self.height - WALL_MARGIN) + 1, COVER_GRID_STEP):
                point = np.array([x, y], dtype=np.float32)
                if self._free(point):
                    pts.append(point)
        return np.asarray(pts, dtype=np.float32).reshape((-1, 2))

    def _find_cover(self):
        if not self.los:
            return self.agent_pos.copy(), True
        distances = np.linalg.norm(self._cover_grid - self.agent_pos, axis=1)
        for i in np.argsort(distances):
            point = self._cover_grid[i]
            if not self.has_line_of_sight(self.opp_pos, point):
                return point.copy(), False
        return self.agent_pos.copy(), False

    def _refresh(self):
        self.los = self.has_line_of_sight(self.agent_pos, self.opp_pos)
        self.cover_point, self.in_cover = self._find_cover()

    def _random_spawn(self, x_lo, x_hi):
        # Search bounded spawn bands first; then use any free point in the arena.
        for lo, hi in ((x_lo, x_hi), (WALL_MARGIN, self.width - WALL_MARGIN)):
            for _ in range(500):
                point = np.array([self.rng.uniform(lo, hi),
                                  self.rng.uniform(WALL_MARGIN, self.height - WALL_MARGIN)],
                                 dtype=np.float32)
                if self._free(point):
                    return point
        raise RuntimeError("Could not find a free entity spawn position")

    def reset(self, rng=None):
        if rng is not None:
            self.rng = rng
        self.agent_pos = self._random_spawn(60, min(200, self.width - WALL_MARGIN))
        self.opp_pos = self._random_spawn(max(400, WALL_MARGIN), self.width - WALL_MARGIN)
        self.agent_health = self.opp_health = 100.0
        self.agent_ammo = AGENT_MAX_AMMO
        self.agent_cooldown = 0
        self.opp_cooldown = int(self.rng.integers(0, OPP_COOLDOWN))
        self.opp_strafe_dir = int(self.rng.choice([-1, 1]))
        self.opp_strafe_timer = int(self.rng.integers(25, 60))
        self.projectiles = []
        self.tick = 0
        self._refresh()

    def distance(self):
        return float(np.linalg.norm(self.opp_pos - self.agent_pos))

    def _spawn_projectile(self, origin, direction, owner_id):
        direction = np.asarray(direction, dtype=np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-5)
        start = origin + direction * MUZZLE_OFFSET
        self.projectiles.append(Projectile(*start, *direction, owner_id=owner_id))

    def nearest_threat(self):
        threats = [np.array([p.x, p.y], dtype=np.float32) - self.agent_pos
                   for p in self.projectiles if p.owner_id == 2 and p.active]
        return min(threats, key=lambda v: float(np.linalg.norm(v))) if threats else None

    def _agent_act(self, action, ev):
        speed = AGENT_SPEED
        moves = {0: (0, -speed), 1: (0, speed), 2: (speed, 0), 3: (-speed, 0)}
        if action in moves:
            self.agent_pos = self._move(self.agent_pos, moves[action])
        elif action == 4:
            if self.agent_ammo > 0 and self.agent_cooldown == 0:
                self.agent_ammo -= 1
                self.agent_cooldown = AGENT_COOLDOWN
                self._spawn_projectile(self.agent_pos, self.opp_pos - self.agent_pos, 1)
                ev["agent_fired"] = True
                ev["fired_blind"] = not self.los
            else:
                ev["invalid_shot"] = True
        elif action == 5 and not self.in_cover:
            delta = self.cover_point - self.agent_pos
            norm = float(np.linalg.norm(delta))
            if norm > 1e-5:
                self.agent_pos = self._move(self.agent_pos, delta * min(1.0, speed / norm))
        elif action == 6:
            delta = self.agent_pos - self.opp_pos
            norm = max(float(np.linalg.norm(delta)), 1e-5)
            moved = self._move(self.agent_pos, delta / norm * speed)
            if np.linalg.norm(moved - self.agent_pos) < 1.0:
                perpendicular = np.array([-delta[1], delta[0]], dtype=np.float32) / norm
                center = np.array([self.width / 2, self.height / 2], dtype=np.float32)
                if np.dot(perpendicular, center - self.agent_pos) < 0:
                    perpendicular = -perpendicular
                moved = self._move(self.agent_pos, perpendicular * speed)
            self.agent_pos = moved

    def _opponent_act(self, ev):
        delta = self.agent_pos - self.opp_pos
        distance = max(float(np.linalg.norm(delta)), 1e-5)
        direction = delta / distance
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
        self.opp_strafe_timer -= 1
        if self.opp_strafe_timer <= 0:
            self.opp_strafe_dir = int(self.rng.choice([-1, 1]))
            self.opp_strafe_timer = int(self.rng.integers(25, 60))
        strafe = perpendicular * self.opp_strafe_dir
        if not self.los or distance > 280:
            move = direction
        elif distance < 140:
            move = -0.7 * direction + 0.7 * strafe
        else:
            move = strafe
        move /= max(float(np.linalg.norm(move)), 1e-5)
        new_pos = self._move(self.opp_pos, move * OPP_SPEED)
        if np.linalg.norm(new_pos - self.opp_pos) < 0.5:
            new_pos = self._move(self.opp_pos, strafe * OPP_SPEED)
        self.opp_pos = new_pos
        if self.los and distance < OPP_MAX_RANGE and self.opp_cooldown == 0:
            angle = float(self.rng.normal(0.0, OPP_AIM_NOISE))
            c, s = np.cos(angle), np.sin(angle)
            aim = np.array([c * direction[0] - s * direction[1],
                            s * direction[0] + c * direction[1]], dtype=np.float32)
            self._spawn_projectile(self.opp_pos, aim, 2)
            self.opp_cooldown = OPP_COOLDOWN
            ev["opp_fired"] = True

    def step(self, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < len(ACTION_NAMES):
            raise ValueError(f"Invalid action {action!r}; expected an integer from 0 to {len(ACTION_NAMES) - 1}")
        action = int(action)
        events = {"agent_fired": False, "fired_blind": False, "invalid_shot": False,
                  "opp_fired": False, "damage_dealt": 0.0, "damage_taken": 0.0}
        self.tick += 1
        self.agent_cooldown = max(0, self.agent_cooldown - 1)
        self.opp_cooldown = max(0, self.opp_cooldown - 1)
        if self.tick % AMMO_REGEN_INTERVAL == 0:
            self.agent_ammo = min(AGENT_MAX_AMMO, self.agent_ammo + 1)
        self._agent_act(action, events)
        self._opponent_act(events)
        for projectile in self.projectiles:
            projectile.update(self.width, self.height, self.obstacles)
            if not projectile.active:
                continue
            if projectile.owner_id == 1 and np.hypot(projectile.x - self.opp_pos[0], projectile.y - self.opp_pos[1]) < HIT_RADIUS:
                projectile.active = False
                dealt = min(AGENT_DAMAGE, self.opp_health)
                self.opp_health -= dealt
                events["damage_dealt"] += dealt
            elif projectile.owner_id == 2 and np.hypot(projectile.x - self.agent_pos[0], projectile.y - self.agent_pos[1]) < HIT_RADIUS:
                projectile.active = False
                taken = min(OPP_DAMAGE, self.agent_health)
                self.agent_health -= taken
                events["damage_taken"] += taken
        self.projectiles = [p for p in self.projectiles if p.active]
        self._refresh()
        return events
