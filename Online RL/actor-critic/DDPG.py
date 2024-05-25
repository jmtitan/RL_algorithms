import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
class ReplayBuffer:
    """
    A simple replay buffer to store experience tuples.
    """
    def __init__(self, capacity, device=torch.device('cuda')):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.counter = 0
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state (array): The current state.
            action (array): The action taken.
            reward (float): The reward received.
            next_state (array): The next state after taking the action.
            done (bool): Whether the episode has ended.
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        self.counter = min(self.counter + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            tuple: Batches of states, actions, rewards, next_states, and dones.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).to(self.device)
        )

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            int: The number of transitions currently stored in the buffer.
        """
        return self.counter

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        super(Actor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        self.action_output = nn.Sequential(
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )
        self.action_lim = action_lim
    def forward(self, state):
        state = torch.clamp(state,-1.1,1.1)
        x = self.extractor(state)
        x = self.action_output(x)
        return self.action_lim * x

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.extractor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        self.value_output = nn.Linear(32, 1)

    def forward(self, state, action):
        state = torch.clamp(state,-1.1,1.1)
        x = torch.cat([state, action], 1)
        x = self.extractor(x)
        x = self.value_output(x)
        return x.squeeze(dim=1)
    
class DDPGAgent:
    """
    The DDPG (Deep Deterministic Policy Gradient) agent.
    """
    def __init__(self, state_dim, action_dim, action_lim, buffer_size=100000, gamma=0.99, tau=0.001, actor_lr=1e-4, critic_lr=1e-3, batch_size=64, device=torch.device('cuda')):
        """
        Initialize the DDPG agent.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            action_lim(int): Zoom size of action output
            buffer_size (int): Maximum size of the replay buffer.
            gamma (float): Discount factor for future rewards.
            tau (float): Soft update parameter.
            actor_lr (float): Learning rate for the actor network.
            critic_lr (float): Learning rate for the critic network.
            batch_size (int): Batch size for sampling from the replay buffer.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim, action_lim).to(device)
        self.target_actor = Actor(state_dim, action_dim, action_lim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.loss = nn.SmoothL1Loss()
        # Initialize replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Perform initial soft update to target networks
        self._soft_update(self.actor, self.target_actor, tau=1.0)
        self._soft_update(self.critic, self.target_critic, tau=1.0)

    def _soft_update(self, local_model, target_model, tau):
        """
        Perform a soft update of the target network parameters.

        Args:
            local_model (nn.Module): The local network.
            target_model (nn.Module): The target network.
            tau (float): The interpolation parameter for the update.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state):
        """
        Select an action given the current state.

        Args:
            state (array): The current state.

        Returns:
            array: The action to take.
        """
        with torch.no_grad():
            action = self.actor(state)
        return action

    def learn(self):
        """
        Train the agent by sampling from the replay buffer and performing gradient descent.
        """
        assert len(self.buffer) > self.batch_size


        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Update Critic
        actions = actions.float()

        next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, next_actions.float())
        q_targets = rewards + (self.gamma * next_q_values * (1 - dones))
        q_values = self.critic(states, actions)
        critic_loss = self.loss(q_values, q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.critic, self.target_critic, self.tau)
        self._soft_update(self.actor, self.target_actor, self.tau)

        return critic_loss.item()
    
    def add_experience(self, state, action, reward, next_state, done):
        """
        Add a new experience to the replay buffer.

        Args:
            state (array): The current state.
            action (array): The action taken.
            reward (float): The reward received.
            next_state (array): The next state after taking the action.
            done (bool): Whether the episode has ended.
        """
        self.buffer.add(state, action, reward, next_state, done)
