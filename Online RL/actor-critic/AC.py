import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AC_net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(AC_net, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # Calculate the output size of the convolutional layers dynamically
        h, w, c = state_dim
        input_dim = (c, h, w)
        conv_output_size = self._get_conv_output(input_dim)
        
        # Fully connected layers for the actor and critic networks
        self.feature = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )

        self.act_layer = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=0)
        )
        self.val_layer = nn.Linear(64, 1)

    def _get_conv_output(self, shape):
        """
        Computes the output size of the convolutional layers given the input shape.
        
        Args:
            shape (tuple): The shape of the input (C, H, W).
        
        Returns:
            int: The size of the output after the convolutional layers.
        """
        with torch.no_grad():
            input_data = torch.zeros(1, *shape)
            output = self.conv_layers(input_data)
            return int(torch.prod(torch.tensor(output.size()[1:])))
    
    def forward(self, x):
        # Extract features using convolutional layers
        x = self.conv_layers(x)
        x = x.flatten()  # Flatten the output of the conv layers

        # Extract features using fully connected layers
        x = self.feature(x)    
        v = self.val_layer(x)           # estimate value of current state

        # For discretized action
        a_prob = self.act_layer(x)      # get current action prob. P(a_i | s_i)
        a_disb = Categorical(a_prob)    # get current action distribution P(a | s_i)
        a = a_disb.sample()             # sample action from the distribution
        a_logprob = a_disb.log_prob(a)
        return v, a.item(), a_logprob




class ActorCritic:

    def __init__(self, state_dim, action_dim, learning_rate=0.01, reward_decay=0.99, device=torch.device('cuda')):

        self.gamma = reward_decay
        self.lr = learning_rate
        self.device = device
        self.net = AC_net(state_dim, action_dim).to(device)
        print(self.net)
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
    
    def trainer(self, task, total_epochs):
        
        env = gym.make(task)    
        S_DIM = env.observation_space.shape[0]
        A_DIM = env.action_space.shape[0]
        A_MAX = env.action_space.high[0]
        
        print(' State Dimensions : ', S_DIM)
        print(' Action Dimensions : ', A_DIM)
        print(' Action Max : ', A_MAX)

        rew_total = []
        for epoch in range(total_epochs):
            obs = env.reset()[0]
            done = False
            final_reward = 0
            while not done:
                
                # env.render()
                loss = 0
                v, a, a_logprob = self.net(torch.FloatTensor(obs).to(device))
                obs_next, _, done, _, info = env.step(a)
                x, x_dot, theta, theta_dot = obs_next

                # modifiying reward function
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                reward = r1 + r2
                
                self.add(v, a_logprob, reward)

                final_reward += reward

                # update network
                if done:
                    loss = self.learn()
                    print('Ep: ', epoch,' | reward:%.3f'%final_reward, ' | loss:%.4f'%loss)
                    rew_total.append(final_reward)
                
                # End condition
                if reward > 1000:
                    break

                obs = obs_next
        return rew_total