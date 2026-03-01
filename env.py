import pygame
import numpy as np
import random

# Global constant definition
WIDTH, HEIGHT = 400, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 120, 255)
RED = (255, 50, 50)
GREEN = (34, 139, 34)


class DodgeEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NeuralEvade: A Deep RL Survival Benchmark")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.reset()

    def reset(self):
        """Reset game core state"""
        self.agent_x = WIDTH // 2 - 20
        self.agent_y = HEIGHT - 40
        self.agent_speed = 10
        self.obs_x = random.randint(15, WIDTH - 15)
        self.obs_y = -15
        self.obs_speed = random.randint(25, 35)
        self.score = 0
        self.lives = 3
        self.done = False
        return self._get_state()

    def _get_state(self):
        """Obtain normalized state vector"""
        state = [self.agent_x / WIDTH, self.obs_x / WIDTH, self.obs_y / HEIGHT, self.obs_speed / 15.0]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """Execute game step logic"""
        if action == 0 and self.agent_x > 0:
            self.agent_x -= self.agent_speed
        elif action == 1 and self.agent_x < WIDTH - 40:
            self.agent_x += self.agent_speed

        self.obs_y += self.obs_speed
        reward = 0.1

        if self.obs_y > HEIGHT:
            self.obs_x = random.randint(15, WIDTH - 15)
            self.obs_y = -15
            self.obs_speed = random.randint(25, 35)
            self.score += 1
            reward = 1.0

        agent_rect = pygame.Rect(self.agent_x, self.agent_y, 40, 20)
        obs_rect = pygame.Rect(self.obs_x - 15, self.obs_y - 15, 30, 30)

        if agent_rect.colliderect(obs_rect):
            self.lives -= 1
            reward = -10.0
            self.obs_y = -15
            self.obs_x = random.randint(15, WIDTH - 15)
            if self.lives <= 0: self.done = True

        return self._get_state(), reward, self.done

    def render(self, ep=None, eps=None, manual_game_over=False):
        """Core rendering function: Integrate HUD and death mask"""
        self.screen.fill(WHITE)
        # Drawing Entity
        pygame.draw.rect(self.screen, BLUE, (self.agent_x, self.agent_y, 40, 20))
        pygame.draw.circle(self.screen, RED, (self.obs_x, int(self.obs_y)), 15)

        # Drawing HUD
        score_txt = self.font.render(f"Score: {self.score}", True, BLACK)
        lives_txt = self.font.render(f"Life: {self.lives}", True, RED)
        self.screen.blit(score_txt, (10, 10))
        self.screen.blit(lives_txt, (10, 35))

        if ep is not None:
            info_txt = self.font.render(f"Ep: {ep}  Eps: {eps:.2f}", True, BLACK)
            self.screen.blit(info_txt, (WIDTH - 140, 10))

        # Death mask logic
        if manual_game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.set_alpha(160)
            overlay.fill((230, 230, 230))
            self.screen.blit(overlay, (0, 0))

            go_font = pygame.font.SysFont("Arial", 48, bold=True)
            hint_font = pygame.font.SysFont("Arial", 22)

            go_txt = go_font.render("GAME OVER", True, RED)
            retry_txt = hint_font.render("Press 'R' or Click to Restart", True, BLACK)

            self.screen.blit(go_txt, (WIDTH // 2 - 110, HEIGHT // 2 - 50))
            self.screen.blit(retry_txt, (WIDTH // 2 - 120, HEIGHT // 2 + 20))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()


def show_start_screen(env):
    """Display the startup menu"""
    waiting = True
    while waiting:
        env.screen.fill(WHITE)
        f_large = pygame.font.SysFont("Arial", 40, bold=True)
        f_small = pygame.font.SysFont("Arial", 20)

        t = f_large.render("DodgeSquare", True, BLACK)
        r1 = f_small.render("Goal: Dodge red obstacles to survive!", True, GREEN)
        r2 = f_small.render("Lives: 3. Controls: Arrow Keys.", True, BLACK)
        h = f_small.render("CLICK SCREEN to Start Game", True, BLUE)

        env.screen.blit(t, (WIDTH // 2 - 100, 180))
        env.screen.blit(r1, (WIDTH // 2 - 140, 280))
        env.screen.blit(r2, (WIDTH // 2 - 130, 310))
        env.screen.blit(h, (WIDTH // 2 - 115, 420))

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()
            if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                waiting = False
        pygame.time.delay(30)


# Manual mode operation entry
if __name__ == "__main__":
    env = DodgeEnv()
    while True:
        show_start_screen(env)  # Display menu
        env.reset()
        manual_game_over = False
        game_active = True

        while game_active:
            action = 2
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); exit()
                if manual_game_over and (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN):
                    if event.type == pygame.MOUSEBUTTONDOWN or event.key == pygame.K_r:
                        game_active = False  # Exit the game loop and return to the outermost display menu

            if not manual_game_over:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]: action = 0
                if keys[pygame.K_RIGHT]: action = 1
                _, _, done = env.step(action)
                if done: manual_game_over = True

            env.render(manual_game_over=manual_game_over)