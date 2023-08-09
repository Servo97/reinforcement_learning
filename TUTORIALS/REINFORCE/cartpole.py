import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

GAMMA = 0.9

class PolicyNetwork(nn.Module):
    def __init__(self, n_obs, n_actions, hidden_size=256, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1)
        return x
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        action = np.random.choice(self.n_actions, p = np.squeeze(probs.detach().numpy()))
        # action = np.argmax(np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[action])
        return action, log_prob

def update_policy(policy_nw, rewards, log_probs):
    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = 0 
        for pw, r in enumerate(rewards[t:]):
            Gt = Gt + GAMMA**pw * r
        discounted_rewards.append(Gt)
    discounted_rewards = torch.tensor(discounted_rewards)
    # normalize discounted rewards
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    policy_nw.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_nw.optimizer.step()

def main():
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    policy_nw = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)

    max_episode_num = 5000
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode  in range(max_episode_num):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        for steps in range(max_steps):
            env.render()
            action, log_prob = policy_nw.get_action(state)
            new_state, reward, done, truncate, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done or truncate:
                update_policy(policy_nw, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode%1 == 0:
                    sys.stdout.write(f"Episode: {episode}, Total Reward: {np.round(np.sum(rewards), decimals = 3)}, Avg Reward: {np.round(np.mean(all_rewards[-10:]), decimals = 3)}, Length: {steps}")
                break

            state = new_state

    plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel('Episode')
    plt.show()

if __name__ == '__main__':
    main()