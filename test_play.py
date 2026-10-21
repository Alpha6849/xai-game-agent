import pygame
from env.gym_wrapper import CombatArenaEnv

pygame.init()
# Keep ammo finite during the manual render demo; a new round restores the starting ammo.
env = CombatArenaEnv(render_mode="human", ammo_regen_interval=None)
obs, info = env.reset()
env.render()

pygame.display.set_caption(
    "XAI Combat Arena | Click window, WASD/Arrows move, Space shoots, C cover, R retreat, Q quit"
)

print("Controls:")
print("Arrow Keys / WASD : Move (North, South, East, West)")
print("Spacebar          : Shoot")
print("C                 : Take Cover")
print("R                 : Retreat")
print("Q                 : Quit")

running = True
last_move_action = 0  # Default facing/action direction
key_actions = {
    pygame.K_w: 0, pygame.K_UP: 0,
    pygame.K_s: 1, pygame.K_DOWN: 1,
    pygame.K_d: 2, pygame.K_RIGHT: 2,
    pygame.K_a: 3, pygame.K_LEFT: 3,
    pygame.K_SPACE: 4,
    pygame.K_c: 5,
    pygame.K_r: 6,
}

while running:
    event_action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
            elif event.key in key_actions:
                event_action = key_actions[event.key]

    if not running:
        break

    keys = pygame.key.get_pressed()
    held_action = None
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        held_action = 0
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        held_action = 1
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        held_action = 2
    elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
        held_action = 3
    elif keys[pygame.K_SPACE]:
        held_action = 4
    elif keys[pygame.K_c]:
        held_action = 5
    elif keys[pygame.K_r]:
        held_action = 6

    action = event_action if event_action is not None else held_action
    if action in (0, 1, 2, 3):
        last_move_action = action

    # Execute move or step idle tick
    step_action = action if action is not None else last_move_action
    obs, reward, terminated, truncated, info = env.step(step_action)

    if terminated or truncated:
        print(f"Round ended! Winner: {info.get('winner')} | Steps: {info.get('step')}")
        obs, info = env.reset()

env.close()
