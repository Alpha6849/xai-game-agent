import pygame
from env.gym_wrapper import CombatArenaEnv

pygame.init()
env = CombatArenaEnv(render_mode="human")
obs, info = env.reset()
env.render()
clock = pygame.time.Clock()

print("Controls:")
print("Arrow Keys / WASD : Move (North, South, East, West)")
print("Spacebar          : Shoot")
print("C                 : Take Cover")
print("R                 : Retreat")
print("Q                 : Quit")

running = True
last_move_action = 0  # Default facing/action direction

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:
        break

    action = None
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        action = 0
        last_move_action = 0
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        action = 1
        last_move_action = 1
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        action = 2
        last_move_action = 2
    elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
        action = 3
        last_move_action = 3
    elif keys[pygame.K_SPACE]:
        action = 4
    elif keys[pygame.K_c]:
        action = 5
    elif keys[pygame.K_r]:
        action = 6

    # Execute move or step idle tick
    step_action = action if action is not None else last_move_action
    obs, reward, terminated, truncated, info = env.step(step_action)

    if terminated or truncated:
        print(f"Round ended! Winner: {info.get('winner')} | Steps: {info.get('step')}")
        obs, info = env.reset()

    clock.tick(30)

env.close()