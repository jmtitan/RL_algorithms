import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    '''
    PolicyNetwork defines the neural network for the policy function.

    Args:
        input_shape (tuple): The shape of the input (observation space).
        num_actions (int): The number of possible actions (action space).

    Outputs:
        logits (Tensor): The logits for each action.
    '''
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Apply ReLU activation
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)  # Output logits
        return logits

class ValueNetwork(nn.Module):
    '''
    ValueNetwork defines the neural network for the value function.

    Args:
        input_shape (tuple): The shape of the input (observation space).

    Outputs:
        value (Tensor): The predicted value of the state.
    '''
    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Apply ReLU activation
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)  # Output value
        return value

class PPO:
    '''
    PPO defines the Proximal Policy Optimization algorithm.

    Args:
        input_shape (tuple): The shape of the input (observation space).
        num_actions (int): The number of possible actions (action space).
        learning_rate (float): The learning rate for the optimizer.
        gamma (float): The discount factor.
        gae_lambda (float): The lambda for GAE (Generalized Advantage Estimation).
        clip_param (float): The clipping parameter for PPO.
        num_steps (int): The number of steps to run for each environment per update.
        batch_size (int): The batch size for PPO updates.
        ppo_epochs (int): The number of epochs for PPO updates.

    Outputs:
        policy_loss.item() (float): The policy loss.
        value_loss.item() (float): The value loss.
        entropy.item() (float): The entropy of the action distribution.
    '''
    def __init__(self, input_shape, num_actions, learning_rate=2.5e-4, gamma=0.99, gae_lambda=0.95, clip_param=0.2, num_steps=128, batch_size=256, ppo_epochs=4):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs

        self.policy_net = PolicyNetwork(input_shape, num_actions).cuda()
        self.value_net = ValueNetwork(input_shape).cuda()
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)

    def compute_gae(self, next_value, rewards, masks, values):
        '''
        Compute GAE (Generalized Advantage Estimation).

        Args:
            next_value (float): The estimated value of the next state.
            rewards (list): The list of rewards.
            masks (list): The list of masks indicating episode boundaries.
            values (list): The list of estimated values.

        Outputs:
            returns (list): The list of computed returns.
        '''
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_update(self, states, actions, log_probs, returns, advantages):
        '''
        Perform a PPO update.

        Args:
            states (Tensor): The batch of states.
            actions (Tensor): The batch of actions.
            log_probs (Tensor): The batch of log probabilities of the actions.
            returns (Tensor): The batch of returns.
            advantages (Tensor): The batch of advantages.

        Outputs:
            policy_loss.item() (float): The policy loss.
            value_loss.item() (float): The value loss.
            entropy.item() (float): The entropy of the action distribution.
        '''
        dist = Categorical(logits=self.policy_net(states))
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = (returns - self.value_net(states)).pow(2).mean()

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        return policy_loss.item(), value_loss.item(), entropy.item()
