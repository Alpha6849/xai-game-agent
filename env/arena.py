import pygame
import numpy as np

class Obstacle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, surface):
        pygame.draw.rect(surface, (120, 120, 120), self.rect)
        pygame.draw.rect(surface, (70, 70, 70), self.rect, 2)


class Projectile:
    def __init__(self, x, y, dx, dy, speed=8.0, owner_id=1):
        self.x = float(x)
        self.y = float(y)
        norm = max(np.hypot(dx, dy), 1e-5)
        self.vx = (dx / norm) * speed
        self.vy = (dy / norm) * speed
        self.owner_id = owner_id
        self.radius = 4
        self.active = True

    def update(self, width, height, obstacles):
        self.x += self.vx
        self.y += self.vy
        if not (0 <= self.x <= width and 0 <= self.y <= height):
            self.active = False
            return
        # Cast to int for pygame.Rect
        proj_rect = pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            int(self.radius * 2),
            int(self.radius * 2),
        )
        for obs in obstacles:
            if obs.rect.colliderect(proj_rect):
                self.active = False
                return

    def draw(self, surface):
        if self.active:
            pygame.draw.circle(surface, (255, 200, 50), (int(self.x), int(self.y)), self.radius)


class ArenaSimulation:
    def __init__(self, width=600, height=600):
        self.width = width
        self.height = height
        self.obstacles = [
            Obstacle(250, 220, 100, 160),  # Central cover
            Obstacle(100, 100, 60, 60),
            Obstacle(440, 440, 60, 60),
        ]
        self.reset()

    def reset(self):
        self.agent_pos = np.array([120.0, 300.0], dtype=np.float32)
        self.opp_pos = np.array([480.0, 300.0], dtype=np.float32)
        self.agent_health = 100.0
        self.opp_health = 100.0
        self.agent_ammo = 30
        self.opp_ammo = 30
        self.projectiles = []

    def perform_action(self, action):
        speed = 5.0
        # 0: North, 1: South, 2: East, 3: West, 4: Shoot, 5: Cover, 6: Retreat
        if action == 0:
            self.agent_pos[1] = max(20.0, self.agent_pos[1] - speed)
        elif action == 1:
            self.agent_pos[1] = min(self.height - 20.0, self.agent_pos[1] + speed)
        elif action == 2:
            self.agent_pos[0] = min(self.width - 20.0, self.agent_pos[0] + speed)
        elif action == 3:
            self.agent_pos[0] = max(20.0, self.agent_pos[0] - speed)
        elif action == 4:
            if self.agent_ammo > 0:
                self.agent_ammo -= 1
                dx = self.opp_pos[0] - self.agent_pos[0]
                dy = self.opp_pos[1] - self.agent_pos[1]
                self.projectiles.append(Projectile(self.agent_pos[0], self.agent_pos[1], dx, dy, owner_id=1))
        elif action == 5:  # Move toward cover
            target = np.array([210.0, 300.0], dtype=np.float32)
            move_dir = target - self.agent_pos
            norm = max(np.hypot(*move_dir), 1e-5)
            self.agent_pos += (move_dir / norm) * speed
        elif action == 6:  # Retreat away from opponent
            move_dir = self.agent_pos - self.opp_pos
            norm = max(np.hypot(*move_dir), 1e-5)
            self.agent_pos += (move_dir / norm) * speed
            self.agent_pos[0] = np.clip(self.agent_pos[0], 20.0, self.width - 20.0)
            self.agent_pos[1] = np.clip(self.agent_pos[1], 20.0, self.height - 20.0)

    def update_projectiles(self):
        hit_occurred = False
        for p in self.projectiles:
            p.update(self.width, self.height, self.obstacles)
            if p.active and p.owner_id == 1:
                dist_to_opp = np.hypot(p.x - self.opp_pos[0], p.y - self.opp_pos[1])
                if dist_to_opp < 18.0:
                    p.active = False
                    self.opp_health = max(0.0, self.opp_health - 20.0)
                    hit_occurred = True
        self.projectiles = [p for p in self.projectiles if p.active]
        return hit_occurred