
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

class AC_net(nn.Module):

    def __init__(self):
        super(AC_net, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(4, 10),
            )
        self.act_layer = nn.Sequential(
            nn.Linear(10, 2),
            nn.Softmax(dim=0)
            )
        self.val_layer = nn.Linear(10, 1)
    
    def forward(self, x):

        x = self.feature(x)    
        v = self.val_layer(x)           # estimate value of current state
        a_prob = self.act_layer(x)      # get current action prob. P(a_i | s_i)
        a_disb = Categorical(a_prob)    # get current action distribution P(a | s_i)
        a = a_disb.sample()             # sample action from the distribution

        a_logprob = a_disb.log_prob(a)
        return v, a.item(), a_logprob

class ActorCritic:

    def __init__(self, learning_rate=0.01, reward_decay=0.99, device=torch.device('cuda')):

        self.gamma = reward_decay
        self.lr = learning_rate
        self.device = device
        self.net = AC_net().to(device)
        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        self.logprobs = []
        self.val_buf = []
        self.rew_buf = []

    def clear(self):
        self.logprobs = []
        self.val_buf = []
        self.rew_buf = []

    def add(self, val, act_log_prob, rew):
        self.val_buf.append(val)
        self.logprobs.append(act_log_prob)
        self.rew_buf.append(rew)

    def computeloss(self):

        # reward decay
        rewards = []
        dis_reward = 0
        for rew in self.rew_buf[::-1]:
            dis_reward = rew + self.gamma * dis_reward
            rewards.insert(0, dis_reward)
        
        # reward nomalization
        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        # loss
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.val_buf, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value[0], reward)
            loss += (action_loss + value_loss)   

        return loss
    
    def learn(self):

        self.optimizer.zero_grad()
        loss = self.computeloss()
        loss.backward()
        self.optimizer.step()
        self.clear()
        return loss.item()