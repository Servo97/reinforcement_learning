import time
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, n_obs, n_actions, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(n_obs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPO:
    def __init__(self, policy_nw, env, hyperparams):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.n_obs = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]

        self.actor = policy_nw(self.n_obs, self.n_actions, hyperparams['hidden_dim']).to(self.device)
        self.critic = policy_nw(self.n_obs, 1, hyperparams['hidden_dim']).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=hyperparams['lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hyperparams['lr'])

        self.covariance_mat = torch.eye(self.n_actions)*0.5
        self.covariance_mat = self.covariance_mat.to(self.device)

        self._init_hyperparams_(hyperparams)
        self.logger = {
            'delta_t': time.time_ns(),
            'timesteps': 0,
            'iterations': 0,
            'batch_lengths': [],
            'batch_rewards': [],
            'actor_losses': [],
        }

    def train(self, timesteps):
        print(f"Learning... Running {self.max_timesteps_per_ep} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {timesteps} timesteps")
        t = 0
        iters = 0
        while t < timesteps:
            b_obs, b_actions, b_log_probs, b_returns, b_lengths = self.collect_batch_experiences()
            t += np.sum(b_lengths)
            iters += 1
            self.logger['timesteps'] += np.sum(b_lengths)
            self.logger['iterations'] += 1

            value, _ = self.evaluate_critic(b_obs, b_actions)
            A_k = b_returns - value.detach()
            # Normalize advantage function for stability
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            for i in range(self.n_updates_per_iteration):
                value, log_probs = self.evaluate_critic(b_obs, b_actions)
                # e^diff of logs == derivative of ratio of probs
                ratio = torch.exp(log_probs - b_log_probs) 
                # Clipped surrogate objective
                # print(ratio.device, A_k.device)
                L = torch.min(ratio*A_k, torch.clamp(ratio, 1-self.clip, 1+self.clip)*A_k)
                actor_loss = -L.mean()
                critic_loss = F.mse_loss(value, b_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                self.logger['actor_losses'].append(actor_loss.cpu().detach())

            self._log_summary()
            if iters % self.save_freq == 0:
                torch.save(self.actor.state_dict(), f"models/ppo_actor_{iters}.pth")
                torch.save(self.critic.state_dict(), f"models/ppo_critic_{iters}.pth")

    def collect_batch_experiences(self):
        b_obs = []
        b_actions = []
        b_log_probs = []
        b_rewards = []
        b_returns = []
        b_lengths = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rewards = []
            obs, _ = self.env.reset()
            for i in range(self.max_timesteps_per_ep):
                # if self.render and self.logger['iterations'] % self.render_every == 0:
                #     self.env._render_frame()
                t += 1
                b_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, reward, terminate, truncate, _ = self.env.step(action.cpu().detach().numpy())
                ep_rewards.append(reward)
                b_actions.append(action)
                b_log_probs.append(log_prob.reshape(1))

                if terminate or truncate:
                    break
            b_rewards.append(ep_rewards)
            b_lengths.append(i+1)
        
        b_obs = torch.tensor(np.array(b_obs), dtype=torch.float)
        b_actions = torch.reshape(torch.cat(b_actions, dim=0), (-1, 1))
        b_log_probs = torch.reshape(torch.cat(b_log_probs, dim=0), (-1, 1))
        b_returns = self.batch_returns(b_rewards)
        self.logger['batch_rewards'] = b_rewards
        self.logger['batch_lengths'] = b_lengths
        return b_obs, b_actions, b_log_probs, b_returns, b_lengths
                    
    def batch_returns(self, b_rewards):
        b_returns = []
        for ep_rewards in reversed(b_rewards):
            discounted_reward = 0
            for reward in reversed(ep_rewards):
                discounted_reward = reward + self.gamma*discounted_reward
                b_returns.append(discounted_reward)
        
        b_returns = torch.tensor(b_returns[::-1], dtype=torch.float).to(self.device)
        return b_returns

    def get_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        else:
            obs = obs.to(self.device)
        mean = self.actor(obs)
        dist = torch.distributions.MultivariateNormal(mean, self.covariance_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_critic(self, obs, actions):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        else:
            obs = obs.to(self.device)
        value = self.critic(obs).squeeze()
        mean = self.actor(obs)
        dist = torch.distributions.MultivariateNormal(mean, self.covariance_mat)
        log_prob = dist.log_prob(actions)
        return value, log_prob

    def _init_hyperparams_(self, hyperparams):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_ep = 1600
        self.n_updates_per_iteration = 5
        self.gamma = 0.99
        self.lr = 3e-4
        self.clip = 0.2
        self.render = True
        self.render_every = 50
        self.save_freq = 10
        self.seed = None

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparams.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['timesteps']
        i_so_far = self.logger['iterations']
        avg_ep_lens = np.mean(self.logger['batch_lengths'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rewards']])
        avg_actor_loss = np.mean([losses.mean().item() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lengths'] = []
        self.logger['batch_rewards'] = []
        self.logger['actor_losses'] = []

def train(env, hyperparams, actor_model, critic_model):
    print("Training", flush=True)
    model = PPO(PolicyNetwork, env, hyperparams)
    if (actor_model is not None) and (critic_model is not None):
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print("Successfully Loaded models", flush=True)
    else:
        print("Training from Scratch", flush=True)

    model.train(timesteps=200_000_000)

if __name__ == "__main__":
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    hyperparams = {
        'timesteps_per_batch': 4800,
        'max_timesteps_per_ep': 1600,
        'n_updates_per_iteration': 5,
        'gamma': 0.99,
        'lr': 3e-4,
        'clip': 0.2,
        'render': True,
        'render_every': 50,
        'save_freq': 10,
        'seed': None,
        'hidden_dim': 128
    }
    train(env, hyperparams, None, None)