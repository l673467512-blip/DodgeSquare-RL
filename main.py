import pygame
import matplotlib.pyplot as plt
from env import DodgeEnv
from agent import DQNAgent


def show_start_screen(env):

    waiting = True
    while waiting:
        env.screen.fill((255, 255, 255))
        # Set font
        font_large = pygame.font.SysFont("Arial", 40, bold=True)
        font_small = pygame.font.SysFont("Arial", 20)

        # Render content
        title = font_large.render("DodgeSquare", True, (0, 0, 0))
        rule1 = font_small.render("Goal: Dodge red obstacles to survive!", True, (34, 139, 34))
        rule2 = font_small.render("The agent has 3 lives per episode.", True, (0, 0, 0))
        start_hint = font_small.render("CLICK SCREEN or PRESS ANY KEY to Start", True, (0, 120, 255))

        # Center-aligned display
        env.screen.blit(title, (60, 180))
        env.screen.blit(rule1, (50, 280))
        env.screen.blit(rule2, (100, 310))
        env.screen.blit(start_hint, (40, 420))

        pygame.display.flip()

        # Handling event streams
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                exit()
            # Compatibility fix: Press any key or click the left mouse button to start
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                waiting = False

        pygame.time.delay(30)  # Maintain response speed


if __name__ == "__main__":
    env = DodgeEnv()
    show_start_screen(env)

    # Initialize a 4-dimensional state space and 3 actions
    agent = DQNAgent(state_size=4, action_size=3)
    scores = []
    EPISODES = 200  # Total training epochs

    print("Training Started...")
    for e in range(EPISODES):
        state = env.reset()  # Initialize the environment
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close();
                    exit()

            action = agent.act(state)  # Select actions based on the ε-greedy strategy
            next_state, reward, done = env.step(action)  # Perform an action and receive a reward
            agent.remember(state, action, reward, next_state, done)  # Store in the experience replay pool
            agent.train()  # Batch learning to update weights

            state = next_state

            # Render the UI display every 10 rounds
            if e % 10 == 0:
                env.render(e + 1, agent.epsilon)

            if done:
                scores.append(env.score)

                # --- New: The exploration rate will only decrease after each round ends ---
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                # ----------------------------------
                if (e + 1) % 10 == 0:
                    print(f"Episode: {e + 1}/{EPISODES} | Last Score: {env.score} | Eps: {agent.epsilon:.2f}")
                break

    env.close()  # Close the window when the game ends

    # Draw and save the training result graph
    plt.plot(scores)
    plt.title("DQN Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig("final_curve.png")
    plt.show()