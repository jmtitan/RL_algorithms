import numpy as np
import torch
from torch import nn
from torch.optim import Adam

class SumTree:
    '''
    Build tree and data,
    Because SumTree has a special data structure,
    Both can be stored using a one-dimensional np.array

    This version decouples sumtree from the transition,
    Unlike the version by Jaromír Janisch and Morvan Python.
    '''
    def __init__(self, capacity):
        '''
        args:
            capacity: the capacity of the root node of the sumtree
        '''
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.ptr = 0

    def add(self, p):
        '''
        Add a new sample to the tree and data
        args:
            p: priority of the sample
        '''
        tree_idx = self.ptr + self.capacity - 1
        self.update(tree_idx, p)
        self.ptr += 1

        if self.ptr >= self.capacity:
            self.ptr = 0

    def update(self, tree_idx, p):
        '''
        Update the tree when a sample has a new TD-error
        args:
            tree_idx: index of the tree node
            p: priority value
        '''
        modify = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:    # Update all nodes along the path
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += modify

    def get_leaf(self, v):
        '''
        Extract a sample based on the selected v point
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
            v: selected point
        '''
        parent_idx = 0
        while True:     # Morvan Python's method, more efficient than recursion
            cl_idx = 2 * parent_idx + 1         # Left leaf node (odd), right (even)
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # Select left if larger than left leaf node, otherwise select right
                leaf_idx = parent_idx
                break
            else:       # Search down, find the node larger than v
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
        Get sum (priorities) (read-only property)
        '''
        return self.tree[0]

class PriorityReplayBuffer:
    '''
    Priority Experience Replay Buffer
    '''

    def __init__(self, N, n_states):
        '''
        Initialize the replay buffer
        args:
            N: capacity
        '''
        self.capacity = N
        self.counter = 0
        self.buf = np.zeros([self.capacity, 2 * n_states + 2])
        self.sumtree = SumTree(self.capacity)

        self.delta = 0.01  # Small number to avoid zero priority
        self.alpha = 0.6  # [0~1] Convert TD_error to priority
        self.beta = 0.4  # Initial value for prioritized sampling, eventually becomes 1
        self.beta_increment_per_sampling = 0.001  # Sampling increment step
        self.abs_err_upper = 1  # Upper limit for abs error clipping

    def add(self, s1, a, r, s2):
        '''
        Add experience
        args:
            s1: observation (np.array, float, len=4)
            a: action (int)
            r: reward (float)
            s2: next observation (np.array, float, len=4)
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
        Priority Experience Sampling
        args:
            n: minibatch size
        '''
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.buf[0].size)), np.empty((n, 1))
        pri_seg = self.sumtree.total_p / n       # Divide priority interval
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.sumtree.tree[-self.sumtree.capacity:]) / self.sumtree.total_p     # for later calculate ISweight

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1) # Divide interval, a and b are interval boundaries
            v = np.random.uniform(a, b)             # Randomly sample a value within the interval
            idx, p, data_idx = self.sumtree.get_leaf(v)
            data = self.buf[data_idx]
            prob = p / self.sumtree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        '''
        Batch update sumtree
        '''
        abs_errors = abs_errors.cpu().detach().numpy() + self.delta # convert to abs and avoid zero
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.sumtree.update(ti, p)

class Net(nn.Module):
    '''
    Neural network for approximating Q-values
    '''
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        # Feature extraction block
        self.feature = nn.Sequential(
            nn.Linear(n_states, 10),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(10, n_actions)
        )

    def forward(self, x):
        '''
        Forward pass through the network
        '''
        x = self.feature(x)
        qvalue = self.classifier(x)
        return qvalue


class PERDQN:
    '''
    DQN Agent with Prioritized Experience Replay
    '''
    def __init__(self, 
                 n_states,
                 n_actions,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 eps_greedy=0.9,
                 net_update_frequency=100,
                 mini_batch=128,
                 replay_buffer=None,
                 device=torch.device('cuda')):
        '''
        Initialize the PERDQN agent.
        
        Args:
            n_states (int): Number of states.
            n_actions (int): Number of actions.
            learning_rate (float): Learning rate for the optimizer.
            reward_decay (float): Discount factor for future rewards.
            eps_greedy (float): Epsilon value for epsilon-greedy policy.
            net_update_frequency (int): Frequency to update target network.
            mini_batch (int): Size of mini-batch for training.
            replay_buffer (object): Instance of the replay buffer.
            device (torch.device): Device to run the model (cpu or cuda).
        '''
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = learning_rate
        self.gamma = reward_decay

        self.eps_begin = 1 - eps_greedy
        self.eps_end = eps_greedy
        self.eps = 0.1
        self.epsilon_decay = 100

        self.update_step = net_update_frequency
        self.learning_step = 0  # To track target network update
        self.device = device

        self.eval_net = Net(n_states, n_actions).to(device)
        self.target_net = Net(n_states, n_actions).to(device)

        self.buffer = replay_buffer
        self.mini_batch = mini_batch

        self.optimizer = Adam(self.eval_net.parameters(), lr=learning_rate)

    def choose_action(self, obs, epoch):
        '''
        Choose an action based on epsilon-greedy policy.
        
        Args:
            obs (array): Current state.
            epoch (int): Current epoch.
        
        Returns:
            int: Action to take.
        '''
        # eps-decay
        self.eps = self.eps_begin + (self.eps_end - self.eps_begin) * np.exp(-epoch / self.epsilon_decay)
        if np.random.uniform() < self.eps:
            return np.random.randint(0, self.n_actions)
        else:
            obs = torch.FloatTensor([obs]).to(self.device)
            q_value = self.target_net(obs)
            return q_value.argmax().cpu().data.numpy()

    def update(self):
        '''
        Update the target Q-network.
        '''
        state_dict = self.eval_net.state_dict()
        self.target_net.load_state_dict(state_dict)

    def criterion(self, q_eval, q_hat, ISWeights, tree_idx):
        '''
        Calculate the loss and update the sumtree of PER.
        
        Args:
            q_eval (Tensor): Estimated Q-values.
            q_hat (Tensor): Target Q-values.
            ISWeights (Tensor): Importance sampling weights.
            tree_idx (array): Indices in the sumtree.
        
        Returns:
            Tensor: Loss value.
        '''
        abs_tderror = torch.abs(q_eval - q_hat)  # Update sumtree
        self.buffer.batch_update(tree_idx, abs_tderror)
        loss = torch.mul(abs_tderror**2, ISWeights).mean()
        return loss

    def learn(self):
        '''
        Learning step to update the networks.
        
        Returns:
            Tensor: Loss value.
        '''
        if self.learning_step % self.update_step == 0:  # Update target network every update_step steps
            self.update()

        tree_idx, batch_memory, ISWeights = self.buffer.sample(self.mini_batch)
        ISWeights = torch.as_tensor(ISWeights / np.max(ISWeights), dtype=torch.float32, device=self.device)

        b_s1 = torch.FloatTensor(batch_memory[:, :self.n_states]).to(self.device)
        b_a = torch.LongTensor(batch_memory[:, self.n_states].astype(int)).to(self.device)
        b_r = torch.FloatTensor(batch_memory[:, self.n_states + 1]).to(self.device)
        b_s2 = torch.FloatTensor(batch_memory[:, self.n_states + 2:]).to(self.device)

        q_eval = self.eval_net(b_s1).gather(1, b_a.unsqueeze(-1)).squeeze(-1)

        q_next = self.target_net(b_s2).max(1)[0].detach()
        q_hat = b_r + self.gamma * q_next

        loss = self.criterion(q_eval, q_hat, ISWeights, tree_idx)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learning_step += 1
        return loss
