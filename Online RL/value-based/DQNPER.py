import numpy as np
import torch
from torch import nn
from torch.autograd.variable import Variable
from torch.optim.adam import Adam
from collections import deque

class SumTree():
    '''
    Build tree 和 data,
    因为 SumTree 有特殊的数据结构,
    所以两者都能用一个一维 np.array 来存储

    Jaromír Janisch的版本以及莫凡python都将transition融入sumtree，
    我将sumtree解耦
    '''
    def __init__(self, capacity):
        '''
        args:
            capacity: sumtree根节点的数据容量
        '''
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.ptr = 0
        
    # 当有新 sample 时, 添加进 tree 和 data
    def add(self, p):
        '''
        当有新 sample 时, 添加进 tree 和 data
        args:
            p: sample的优先级
        '''

        tree_idx = self.ptr + self.capacity - 1
        self.update(tree_idx, p)
        self.ptr += 1

        if self.ptr >= self.capacity:
            self.ptr = 0


    def update(self, tree_idx, p):
        '''
        当 sample 被 train, 有了新的 TD-error, 就在 tree 中更新
        args:
            tree_idx: 树节点下标
            p: 优先级数据
        '''

        modify = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:    # 循环更改该路径所有涉及节点
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += modify

    def get_leaf(self, v):
        '''
        根据选取的 v 点抽取样本
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]

        args:
            v: 选取点
        '''
        parent_idx = 0
        while True:     # 莫凡python的方法，比递归效率更高
            cl_idx = 2 * parent_idx + 1         # 左叶子节点（奇数） 右（偶数）
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # 比左叶子节点大则选择左， 否则选右
                leaf_idx = parent_idx
                break
            else:       # 向下搜索，找到比 v 大的节点
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_p(self):
        '''
        获取 sum (priorities) (只读属性)
        '''
        return self.tree[0]

class Priority_Replay_buffer:
    '''
    优先经验采样池
    '''

    def __init__(self, N, n_states):
        '''
        经验池初始化
        args:
            N: 容量
        '''
        self.capacity = N
        self.counter = 0
        self.buf = np.zeros([self.capacity, 2*n_states+2])
        self.sumtree = SumTree(self.capacity)

        self.delta = 0.01  # 小数避免优先级为0
        self.alpha = 0.6 # [0~1] 将TD_error转换为优先级
        self.beta = 0.4  # 优先级采样初始值， 最终变为1
        self.beta_increment_per_sampling = 0.001 #采样上升幅度
        self.abs_err_upper = 1  # abs error剪枝上限

    def add(self, s1, a, r, s2):
        '''
        经验获取
        args:
            s1: obs (np.array, float, len=4)
            a:  action (int)
            r:  reward (float)
            s2: obs_next (np.array, float, len=4)
        '''
        self.counter += 1
        idx = self.counter % self.capacity
        self.buf[idx] = np.hstack((s1, [a, r], s2))

        max_p = np.max(self.sumtree.tree[-self.sumtree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.sumtree.add(max_p)

    def sample(self, n):
        '''
        优先经验采样
        args:
            n: minibatch
        '''
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.buf[0].size)), np.empty((n, 1))
        pri_seg = self.sumtree.total_p / n       # 优先级区间分割
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.sumtree.tree[-self.sumtree.capacity:]) / self.sumtree.total_p     # for later calculate ISweight

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1) # 按区间划分， a、b为区间上下限
            v = np.random.uniform(a, b)             #在区间随机采样一个值
            idx, p, data_idx = self.sumtree.get_leaf(v)
            data = self.buf[data_idx]
            prob = p / self.sumtree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        '''
        sumtree批量更新
        '''
        abs_errors = abs_errors.cpu().detach().numpy() + self.delta # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.sumtree.update(ti, p)


class Net(nn.Module):
    def __init__(self,n_states, n_actions):
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


class PERDQN:
    
    def __init__(self, 
                n_states,
                n_actions,
                learning_rate=0.01,
                reward_decay=0.9,
                eps_greedy=0.9,
                net_update_frequncy=100,
                mini_batch=128,
                replayer_buffer=None,
                device = torch.device('cuda')):

        self.n_actions = n_actions
        self.n_states = n_states
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

        self.optimizer = Adam(self.eval_net.parameters(), lr=learning_rate)
    
    def choose_action(self, obs, epoch):
        '''
        动作选择
        '''
        #eps-decay
        self.eps = self.eps_begin + (self.eps_end - self.eps_begin) * np.exp(-epoch / self.epsilon_decay)
        if np.random.uniform() < self.eps:
            return np.random.randint(0, self.n_actions)

        else:
            obs = Variable(torch.FloatTensor([obs])).to(self.device)
            q_value = self.target_net(obs)
            return q_value.argmax().cpu().data.numpy()

    def update(self):
        '''
        目标Q网络更新
        '''
        state_dict = self.eval_net.state_dict()
        self.target_net.load_state_dict(state_dict)

    def criterion(self, q_eval, q_hat, ISWeights, tree_idx):
        '''
        计算loss同时更新PER的sumtree
        '''
        abs_tderror = torch.abs(q_eval - q_hat) # 更新sumtree
        self.buffer.batch_update(tree_idx, abs_tderror)
        loss = torch.mul(abs_tderror**2, ISWeights).mean()
        return loss

    def learn(self):
        '''
        学习步骤
        '''
        if self.learning_step % self.update_step == 0:  #每update_step步更新targetnet
            self.update()

        tree_idx, batch_memory, ISWeights = self.buffer.sample(self.mini_batch)
        ISWeights = torch.as_tensor(ISWeights / np.max(ISWeights),dtype=torch.float32,device=self.device)

        b_s1 = Variable(torch.FloatTensor(batch_memory[:, :self.n_states])).to(self.device)
        b_a = Variable(torch.LongTensor(batch_memory[:, self.n_states].astype(int))).to(self.device)
        b_r = Variable(torch.FloatTensor(batch_memory[:, self.n_states+1])).to(self.device)
        b_s2 = Variable(torch.FloatTensor(batch_memory[:, self.n_states+2:])).to(self.device)

        q_eval = self.eval_net(b_s1).gather(1, b_a.unsqueeze(-1)).squeeze(-1)

        q_next = self.target_net(b_s2).max(1)[0].detach()
        q_hat = b_r + self.gamma * q_next

        loss = self.criterion(q_eval, q_hat, ISWeights, tree_idx)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learning_step += 1
        return loss

        