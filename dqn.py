import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.epsilon = env_config["eps_start"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        self.env_name = env_config['name']

        if self.env_name == 'Pong-v0':
            self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
            self.fc1 = nn.Linear(3136, 512)
            self.fc2 = nn.Linear(512, self.n_actions)
        elif self.env_name == 'CartPole-v0':
            self.fc1 = nn.Linear(4, 256)
            self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        # In order to use the Pong_3000.pt model, comment everything out in this function except the lines under the first if-statement
        if self.env_name == 'Pong-v0': 
            x = self.relu(self.conv1(x)) 
            x = self.relu(self.conv2(x))
            x = self.flatten(self.relu(self.conv3(x)))
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        elif self.env_name == 'CartPole-v0':
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        
        batch_size = observation.shape[0]
        
        if (exploit or random.random() > self.epsilon):
            with torch.no_grad():
                action_values = self.forward(observation)
                _, actions = torch.max(action_values, dim=1)
                chosen_action = torch.reshape(actions, (batch_size, 1))
        else:
            chosen_action =  torch.randint(0, self.n_actions, (batch_size, 1)).to(device)

        if (self.epsilon > self.eps_end):
            self.epsilon -= (self.eps_start - self.eps_end)/self.anneal_length
    
        return chosen_action


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # Sample a batch from replay memory
    (obs, action, next_obs, reward) = memory.sample(dqn.batch_size)
    obs = torch.stack(obs).squeeze() # 32 x 4
    action = torch.stack(action).squeeze(1) # 32 x 1
    non_terminal_next_obs = [s for s in next_obs if s is not None]
    non_terminal_next_obs = torch.stack(non_terminal_next_obs).squeeze() # 32 or less x 4
    reward = torch.stack(reward) # 32
    
    # Compute the current estimates of the Q-values for each state-action pair
    q_values = torch.gather(dqn.forward(obs), 1, action)
    
    # Compute the Q-value targets for non-terminal transitions
    non_terminal_indicies = []
    for i in range(dqn.batch_size):
        if next_obs[i] != None:
            non_terminal_indicies.append(i)
    non_terminal_indicies = torch.from_numpy(np.array(non_terminal_indicies)).to(device).long()        

    q_value_targets = torch.zeros((dqn.batch_size)).to(device)
    q_value_targets[non_terminal_indicies] = target_dqn.forward(non_terminal_next_obs).max(1)[0]
    q_value_targets = q_value_targets*dqn.gamma + reward

    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
