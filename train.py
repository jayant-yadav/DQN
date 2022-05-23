import argparse

import gym
import torch
import torch.nn as nn
import numpy as np

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0', 'Pong-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=10, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'Pong-v0': config.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    try:
        # Try to load a previous model
        checkpoint = torch.load(f'models/checkpoint/{args.env}_checkpoint.pt')
        dqn.load_state_dict(checkpoint['state_dict'])
        target_dqn.load_state_dict(checkpoint['target_dqn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_episode = checkpoint['episode']
        scores = checkpoint['scores']
        reward_epi = checkpoint['reward_epi']
        #loss_epi = checkpoint['loss_epi']
        memory = checkpoint['memory']
        best_mean_return = checkpoint['best_mean_return']
        dqn.epsilon = checkpoint['epsilon']
        print(f"Model successfully loaded, episode {start_episode}, best return {best_mean_return}, epsilon {dqn.epsilon}")
    except:
        # Initialize a new model
        start_episode = 0
        # Initianize scores/rewards every episode/mean loss every episode records
        scores = []
        reward_epi = []
        #loss_epi = []
        # Create replay memory.
        memory = ReplayMemory(env_config['memory_size'])
        # Keep track of best evaluation mean return achieved so far.
        best_mean_return = -float("Inf")
        print("New model initialized")


    obs_stack_size = env_config['observation_stack_size']

    steps = 0
    for episode in range(start_episode, env_config['n_episodes']+1):
        done = False

        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        obs_stack = torch.cat(obs_stack_size * [obs]).unsqueeze(0).to(device)

        total_episode_reward = 0
        
        while not done:
            # TODO: Get action from DQN.
            action = dqn.act(obs_stack).to(device)
            
            # Act in the true environment.
            next_obs, reward, done, info = env.step(action.item()+2)

            total_episode_reward += reward

            # Preprocess incoming observation.
            if not done:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
                next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
            else:
                next_obs_stack = None #torch.full((1,4), np.nan).to(device)    
   
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            reward = torch.tensor(reward, device=device)
            memory.push(obs_stack, action, next_obs_stack, reward)
            obs_stack = next_obs_stack # no need to clone as next is redefined
            
            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if (steps%env_config["train_frequency"] == 0):
                optimize(dqn, target_dqn, memory, optimizer)

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if (steps%env_config["target_update_frequency"] == 0):
                target_dqn.load_state_dict(dqn.state_dict())
                
            steps += 1

            
        # Evaluate the current agent.
        reward_epi.append(total_episode_reward)
        print(f"Episode {episode}, reward {total_episode_reward}")
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            scores.append(mean_return)
            
            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            checkpoint = {
                'episode': episode+1,
                'state_dict': dqn.state_dict(),
                'target_dqn_state_dict': target_dqn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scores': scores,
                'reward_epi': reward_epi,
                #'loss_epi': loss_epi,
                'memory': memory,
                'best_mean_return': best_mean_return,
                'epsilon': dqn.epsilon
                }

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
                checkpoint['best_mean_return'] = best_mean_return
                torch.save(checkpoint, f'models/checkpoint_best/{args.env}_checkpoint_best.pt')
            torch.save(checkpoint, f'models/checkpoint/{args.env}_checkpoint.pt')
        
    # Close environment after training is completed.
    env.close()
    plt.figure(1)
    plt.plot(range(0, env_config['n_episodes']+1, args.evaluate_freq), scores)
    plt.xlabel("Number of Episode")
    plt.ylabel("Mean Score");
    plt.title("Evaluation Scores Every 25 Episodes")
    plt.savefig(f'figures/Evaluation_reward.jpg')

    plt.figure(2)
    plt.plot(range(0, env_config['n_episodes']+1, 1), reward_epi)
    plt.xlabel("Number of Episode")
    plt.ylabel("Score");
    plt.title("Scores of Every Episode")
    plt.savefig(f'figures/Every_reward.jpg')
