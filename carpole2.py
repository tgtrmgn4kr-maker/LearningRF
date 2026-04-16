import gymnasium as gym

env = gym.make("CartPole-v0")

# 跑 200 個 episode，每個 episode 都是一次任務嘗試
for i_episode in range(200):
    observation = env.reset()  # 讓 environment 重回初始狀態
    rewards = 0.0  # 累計各 episode 的 reward
    for t in range(250):  # 設個時限，每個 episode 最多跑 250 個 action
        env.render()  # 呈現 environment

        # Key section
        action = env.action_space.sample()  # 在 environment 提供的 action 中隨機挑選
        observation, reward, done, info, _ = env.step(
            action
        )  # 進行 action，environment 返回該 action 的 reward 及前進下個 state

        rewards += float(reward)  # 累計 reward

        if done:  # 任務結束返回 done = True
            print(
                "Episode finished after {} timesteps, total rewards {}".format(
                    t + 1, rewards
                )
            )
            break

env.close()
