import gym
import torch
import random
from AC import ActorCritic
from DDPG import DDPGAgent
from PPO import PPO
from torch.distributions import Categorical


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, gym_name, total_epochs, epsilon, render_mode) -> None:
        self.task = gym_name
        self.env = gym.make(gym_name, render_mode=render_mode)
        self.total_epochs = total_epochs
        self.epsilon = epsilon
        self.reward_list = []
        self.S_DIM = self.env.observation_space.shape
        self.A_DIM = self.env.action_space.shape
        
    def run(self):
        pass

class AC_Trainer(Trainer):
    def __init__(self, gym_name, total_epochs, epsilon=0.7, render_mode="rgb_array") -> None:
        super().__init__(gym_name, total_epochs, epsilon, render_mode)
   
        self.A_DIM = self.env.action_space.n
        print(' State Dimensions : ', self.S_DIM)
        print(' Action Dimensions : ', self.A_DIM)

        self.agent = ActorCritic(state_dim=self.S_DIM, 
                    action_dim=self.A_DIM, 
                    learning_rate=0.01, 
                    reward_decay=0.99, 
                    device=device)

    def run(self):
        for epoch in range(self.total_epochs):
            obs = self.env.reset()[0]
            done = False
            epoch_reward = 0
            steps = 0
            while not done:
                loss = 0
                state_input = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
                v, a, a_logprob = self.agent.net(state_input)
                obs_next, reward, done, _, info = self.env.step(a)
                self.agent.add(v, a_logprob, reward)

                epoch_reward += reward

                obs = obs_next
                steps += 1

                # end condition
                if done:
                    loss = self.agent.learn()
                    print('Ep: ', epoch,' | reward:%.3f'%epoch_reward, ' | loss:%.4f'%loss)
                    self.reward_list.append(epoch_reward)


class DDPG_Trainer(Trainer):
    def __init__(self, gym_name, total_epochs, epsilon=0.7,render_mode="rgb_array") -> None:
        super().__init__(gym_name, total_epochs, epsilon, render_mode)


        self.S_DIM = self.env.observation_space.shape[0]
        self.A_DIM = self.env.action_space.shape[0]
        print(' State Dimensions : ', self.S_DIM)
        print(' Action Dimensions : ', self.A_DIM)
        
        self.A_LIM = self.env.action_space.high[0]

        self.start_learn = 10001
        self.agent = DDPGAgent(state_dim=self.S_DIM, 
                    action_dim=self.A_DIM,
                    action_lim=self.A_LIM)

    def run(self):
        for epoch in range(self.total_epochs):
            obs = self.env.reset()[0]
            done = False
            epoch_reward = 0
            self.epsilon **= 1.1 
            while not done:
                loss = 0
                state_input = torch.FloatTensor(obs).to(device)
                if random.random() > self.epsilon:
                    a = self.agent.select_action(state_input).cpu().numpy()
                else:
                    a = self.env.action_space.sample()
                obs_next, reward, done, _, info = self.env.step(a)
                self.agent.add_experience(obs, a, reward, obs_next, done)

                epoch_reward += reward

                obs = obs_next

                # end condition
                if done:
                    loss = self.agent.learn()
                    print('Ep: ', epoch,' | reward:%.3f'%epoch_reward, ' | loss:%.4f'%loss)
                    self.reward_list.append(epoch_reward)

                if len(self.agent.buffer) > self.start_learn:
                    self.agent.learn()



class PPO_trainer(Trainer):
    def __init__(self, gym_name, total_epochs, epsilon=0.7, render_mode="rgb_array") -> None:
        super().__init__(gym_name, total_epochs, epsilon, render_mode)

        agent = PPO(state_dim=self.S_DIM, 
                    action_dim=self.A_DIM)
        
        self.agent = agent

    def run(self, total_timesteps):
        state = self.env.reset()
        state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)

        for frame_idx in range(1, total_timesteps + 1):
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(self.agent.num_steps):
                dist = Categorical(logits=self.agent.policy_net(state))
                value = self.agent.value_net(state)
                action = dist.sample()

                next_state, reward, done, _ = self.env.step(action.cpu().numpy())
                next_state = torch.FloatTensor(next_state).permute(2, 0, 1).unsqueeze(0).to(device)
                reward = torch.FloatTensor([reward]).to(device)
                mask = torch.FloatTensor([1 - done]).to(device)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append(mask)
                states.append(state)
                actions.append(action)

                state = next_state

                if done:
                    state = self.env.reset()
                    state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)

            next_value = self.agent.value_net(state).detach()
            returns = self.agent.compute_gae(next_value, rewards, masks, values)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantages = returns - values

            for _ in range(self.agent.ppo_epochs):
                self.agent.ppo_update(states, actions, log_probs, returns, advantages)

            if frame_idx % 1000 == 0:
                print(f"Frames: {frame_idx}")