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

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(self.relu(self.conv3(x)))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.

        #print(observation.shape)
        #print(observation)
        #print("\n")
        
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

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    (obs, action, next_obs, reward) = memory.sample(dqn.batch_size)

    obs = torch.stack(obs).squeeze() # 32 x 4
    action = torch.stack(action).squeeze(1) # 32 x 1
    non_terminal_next_obs = [s for s in next_obs if s is not None]
    non_terminal_next_obs = torch.stack(non_terminal_next_obs).squeeze() # 32 or less x 4
    reward = torch.stack(reward) # 32

    
    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    q_values = torch.gather(dqn.forward(obs), 1, action)
    
    
    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    # Stolen from Marks code
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
