import pygame
import matplotlib.pyplot as plt
from env import DodgeEnv
from agent import DQNAgent


def show_start_screen(env):
    """高度兼容的启动界面：支持按任意键或点击鼠标开始"""
    waiting = True
    while waiting:
        env.screen.fill((255, 255, 255))
        # 设置字体
        font_large = pygame.font.SysFont("Arial", 40, bold=True)
        font_small = pygame.font.SysFont("Arial", 20)

        # 渲染内容 (参考样例 UI 设计 [cite: 32])
        title = font_large.render("DodgeSquare", True, (0, 0, 0))
        rule1 = font_small.render("Goal: Dodge red obstacles to survive!", True, (34, 139, 34))
        rule2 = font_small.render("The agent has 3 lives per episode.", True, (0, 0, 0))
        start_hint = font_small.render("CLICK SCREEN or PRESS ANY KEY to Start", True, (0, 120, 255))

        # 居中显示
        env.screen.blit(title, (60, 180))
        env.screen.blit(rule1, (50, 280))
        env.screen.blit(rule2, (100, 310))
        env.screen.blit(start_hint, (40, 420))

        pygame.display.flip()

        # 处理事件流
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                exit()
            # 兼容性修复：按下任意键或点击鼠标左键均可开始
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                waiting = False

        pygame.time.delay(30)  # 保持响应速度


if __name__ == "__main__":
    env = DodgeEnv()
    show_start_screen(env)

    # 初始化 4 维状态空间和 3 个动作 [cite: 17, 18, 46]
    agent = DQNAgent(state_size=4, action_size=3)
    scores = []
    EPISODES = 200  # 总训练轮次 [cite: 53]

    print("Training Started...")
    for e in range(EPISODES):
        state = env.reset()  # 初始化环境 [cite: 21]
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close();
                    exit()

            action = agent.act(state)  # 依据 ε-greedy 策略选择动作 [cite: 49]
            next_state, reward, done = env.step(action)  # 执行动作并获取奖励 [cite: 54, 63]
            agent.remember(state, action, reward, next_state, done)  # 存入经验回放池 [cite: 54]
            agent.train()  # 批量学习更新权重 [cite: 55]

            state = next_state

            # 每 10 局渲染一次画面展示 UI [cite: 11]
            if e % 10 == 0:
                env.render(e + 1, agent.epsilon)

            if done:
                scores.append(env.score)

                # --- 新增：每局结束后才减少探索率 ---
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                # ----------------------------------
                if (e + 1) % 10 == 0:
                    print(f"Episode: {e + 1}/{EPISODES} | Last Score: {env.score} | Eps: {agent.epsilon:.2f}")
                break

    env.close()  # 游戏结束关闭窗口 [cite: 16]

    # 绘制并保存训练结果图
    plt.plot(scores)
    plt.title("DQN Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig("final_curve.png")
    plt.show()