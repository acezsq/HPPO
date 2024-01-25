import gym
import gym_hybrid
env = gym.make('Moving-v0')
obs = env.reset()
for _ in range(1000):
    # env.render()  # 显示图形界面
    action = env.action_space.sample()   # 从动作空间中随机选取一个动作
    print(action)
    # print('*******')
    observation, reward, done, info = env.step(action)  # 用于提交动作，括号内是具体的动作
    # print(observation)
env.close()


