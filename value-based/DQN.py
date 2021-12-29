import numpy as np
import torch
import random
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from collections import deque

class Replay_buffer:

    def __init__(self, N):

        self.capacity = N
        self.counter = 0
        self.buf = deque(maxlen=N)

    def add(self, s1, a, r, s2):

        self.counter += 1
        transition = [s1, a, r, s2]
        self.buf.append(transition)

    def sample(self, minibatch):

        batch  = random.sample(self.buf, minibatch)
        b_s1 = [t[0] for t in batch]
        b_a = [t[1] for t in batch]
        b_r = [t[2] for t in batch]
        b_s2 = [t[3] for t in batch]
        return b_s1, b_a, b_r, b_s2



class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        # 特征提取块
        self.feature = nn.Sequential(
            nn.Linear(n_states, 10),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(10, n_actions)
        )

    def forward(self, x):

        x = self.feature(x)
        qvalue = self.classifier(x)

        return qvalue


class DQN:
    
    def __init__(self, 
                n_states,
                n_actions,
                learning_rate=0.01,
                reward_decay=0.9,
                eps_greedy=0.9,
                net_update_frequncy=100,
                mini_batch=128,
                replayer_buffer=Replay_buffer(N=2000),
                device = torch.device('cuda')):

        self.n_actions = n_actions
        self.lr= learning_rate
        self.gamma = reward_decay
        
        self.eps_begin = 1 - eps_greedy
        self.eps_end = eps_greedy
        self.eps = 0.1
        self.epsilon_decay = 100

        self.update_step = net_update_frequncy
        self.learning_step = 0 # 判断更新target net
        self.device = device

        self.eval_net = Net(n_states, n_actions).to(device)
        self.target_net = Net(n_states, n_actions).to(device)

        self.buffer = replayer_buffer
        self.mini_batch = mini_batch

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.eval_net.parameters(), lr=learning_rate)
    
    def choose_action(self, obs, epoch):
        
        #eps-decay
        self.eps = self.eps_begin + (self.eps_end - self.eps_begin) * np.exp(-epoch / self.epsilon_decay)
    
        if np.random.uniform() < self.eps:
            return np.random.randint(0, self.n_actions)

        else:
            obs = Variable(torch.FloatTensor([obs])).to(self.device)
            q_value = self.target_net(obs)
            return q_value.argmax().item()

    def update(self):
        state_dict = self.eval_net.state_dict()
        self.target_net.load_state_dict(state_dict)

    def learn(self):

        if self.learning_step % self.update_step == 0:  #每update_step步更新targetnet
            self.update()

        b_s1, b_a, b_r, b_s2 = self.buffer.sample(self.mini_batch)

        b_s1 = Variable(torch.FloatTensor(b_s1)).to(self.device)
        b_a = Variable(torch.LongTensor(b_a)).to(self.device)
        b_r = Variable(torch.FloatTensor(b_r)).to(self.device)
        b_s2 = Variable(torch.FloatTensor(b_s2)).to(self.device)

        q_eval = self.eval_net(b_s1).gather(1, b_a.unsqueeze(-1)).squeeze(-1)

        q_next = self.target_net(b_s2).max(1)[0].detach()
        q_hat = b_r + self.gamma * q_next

        loss = self.criterion(q_eval, q_hat)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learning_step += 1
        return loss