import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, n_actions):
        super(PolicyNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(4, 10),
            nn.LeakyReLU(),
            )
        self.act_out = nn.Linear(10, n_actions)
        self.act_prob = nn.Softmax(dim=0)


    def forward(self, x):
        x = self.feature(x)
        all_act = self.act_out(x)
        all_act_prob = self.act_prob(all_act)

        return all_act, all_act_prob

class PolicyGradient():
    '''
    策略梯度法
    '''
    def __init__(self, n_action, learning_rate=0.01, reward_decay=0.95, device=torch.device('cuda')):
        
        self.n_actions = n_action   # n_actions 表示动作的
        self.lr = learning_rate
        self.gamma = reward_decay
        self.net = PolicyNet(n_action).to(device)
        self.optimizer = Adam(self.net.parameters(), lr=learning_rate)
        self.learning_step = 0
        self.ep_obs = []
        self.ep_act = []
        self.ep_rew = []
        self.device = device
        
    def one_hot(self, batch_size, class_num, X):
        return torch.zeros([batch_size, class_num]).to(self.device).scatter_(1, X, 1)

    def loss(self, all_act, all_act_prob, label_act, act_val):

        loss = 0
        criteria = nn.CrossEntropyLoss()
        for act, label, act_v in zip(all_act, label_act, act_val):
            neg_log_prob = criteria(act.unsqueeze(0), label.unsqueeze(0))
            loss += neg_log_prob * act_v
        # neg_log_prob = torch.sum(-torch.log(all_act_prob) * self.one_hot(all_act_prob.shape[0], all_act_prob.shape[1], label_act.unsqueeze(1)), axis=1)
        loss /= label_act.shape[0]

        return loss
    
    def choose_action(self, obs):

        with torch.no_grad():
            _, prob_weights = self.net(Variable(torch.FloatTensor(obs)).to(self.device))
            prob_weights = prob_weights.cpu().numpy()
            action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights.ravel())

        return action
    
    def add_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_act.append(a)
        self.ep_rew.append(r)

    def compute_norm_rewards(self):
        ep_rs = np.zeros_like(self.ep_rew)
        momentum = 0
        for t in reversed(range(0, len(self.ep_rew))):
            momentum = momentum * self.gamma + self.ep_rew[t]
            ep_rs[t] = momentum
        
        ep_rs = ep_rs - np.mean(ep_rs) # reward归一化
        ep_rs = ep_rs / np.std(ep_rs)
        return ep_rs

    def learn(self):

        # discount and normalize episode reward
        ep_rs = self.compute_norm_rewards()

        # 训练一回合
        b_obs = Variable(torch.FloatTensor(self.ep_obs)).to(self.device)
        b_act = Variable(torch.LongTensor(self.ep_act)).to(self.device)
        b_rew = Variable(torch.FloatTensor(ep_rs)).to(self.device)

        all_act, all_act_prob = self.net(b_obs)
        loss = self.loss(all_act, all_act_prob, b_act, b_rew)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learning_step += 1

        self.ep_obs, self.ep_act, self.ep_rew = [], [], []    # empty episode data
        return loss
