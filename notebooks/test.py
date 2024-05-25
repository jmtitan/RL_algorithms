import gym
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym

sys.path.append('.')
sys.path.append('../Online RL/actor-critic')


torch.manual_seed(28)
EPOCHS = 1000
task = "ALE/Pong-v5"

env = gym.make('PongNoFrameskip-v4', render_mode='human')  # 使用正确的环境名称
env.reset()
for _ in range(10):
    env.render()
    env.step(env.action_space.sample())  # 随机采取动作
env.close()