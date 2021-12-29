import numpy as np
import pandas as pd

class Qlearning():

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, mode='train'):
        '''初始化'''
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.eps = 0
        self.eps_end = e_greedy
        self.eps_begin = 1 - e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # qtabel:存储q值的表格，行为状态，列为动作名，内容为动态规划计算的Q值
        self.mode = mode

    def choose_action(self, obs):
        '''策略更新，选择动作'''

        self.check_state_exist(obs)
        # epsilon-greedy
        if self.mode == 'train' and np.random.uniform() < self.eps:    # samle(随机)
            action = np.random.choice(self.actions)
        else:
            state_action = np.array(self.q_table.loc[obs, :])   # sample(agent) / predict
            action = state_action.argmax()

        return action

    def learn(self, state, action, reward, state_, done):
        '''
        策略更新
        args:
            state   当前状态 = obs
            action  原策略选择动作
            reward  原策略选择动作的奖赏
            state_  原策略动作影响得到的下一个状态
            done    是否为策略状态的最后一帧
        '''

        self.check_state_exist(state_)
        q_pred = self.q_table.loc[state, action]
        
        if done:
            self.q_table.loc[state, action] += self.lr * (reward - q_pred)
        else:
            self.q_table.loc[state, action] += self.lr * (reward + self.gamma * self.q_table.loc[state_].max() - q_pred)

    def check_state_exist(self, state):
        '''检测qtable是否存在观测状态'''

        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )