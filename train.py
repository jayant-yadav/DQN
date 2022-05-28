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
    env_config = ENV_CONFIGS[args.env]
    if args.env == 'Pong-v0':
        env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
        obs_stack_size = env_config['observation_stack_size']

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    target_dqn = DQN(env_config=env_config).to(device)

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
        memory = checkpoint['memory']
        best_mean_return = checkpoint['best_mean_return']
        dqn.epsilon = checkpoint['epsilon']
        print(f"Model successfully loaded, episode {start_episode}, best return {best_mean_return}, epsilon {dqn.epsilon}")
    except:
        # Initialize a new model
        start_episode = 0
        scores = []
        reward_epi = []
        memory = ReplayMemory(env_config['memory_size'])
        best_mean_return = -float("Inf")
        print("New model initialized")

    steps = 0
    for episode in range(start_episode, env_config['n_episodes']+1):
        done = False
        total_episode_reward = 0
        
        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        if args.env == 'Pong-v0':
            obs_stack = torch.cat(obs_stack_size * [obs]).unsqueeze(0).to(device)
        
        while not done:
            # Get action from DQN and act in true environment.
            if args.env == 'Pong-v0':
                action = dqn.act(obs_stack).to(device)
                next_obs, reward, done, info = env.step(action.item()+2)
            elif args.env == 'CartPole-v0':
                action = dqn.act(obs).to(device)
                next_obs, reward, done, info = env.step(action.item())
            
            total_episode_reward += reward

            # Preprocess incoming observation.
            if not done:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
                if args.env == 'Pong-v0':
                    next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
            else:
                next_obs = None
                if args.env == 'Pong-v0':
                    next_obs_stack = None 
   
            # Add the transition to the replay memory.
            reward = torch.tensor(reward, device=device)
            
            if args.env == 'Pong-v0':
                memory.push(obs_stack, action, next_obs_stack, reward)
                obs_stack = next_obs_stack # no need to clone as next is redefined
            elif args.env == 'CartPole-v0':
                memory.push(obs, action, next_obs, reward)
                obs = next_obs    
            
            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if (steps%env_config["train_frequency"] == 0):
                optimize(dqn, target_dqn, memory, optimizer)

            # Update the target network every env_config["target_update_frequency"] steps.
            if (steps%env_config["target_update_frequency"] == 0):
                target_dqn.load_state_dict(dqn.state_dict())
                
            steps += 1
            steps = steps%env_config["target_update_frequency"]

            
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

    # Plot evaluation returns
    plt.figure(1)
    plt.plot(range(0, env_config['n_episodes']+1, args.evaluate_freq), scores)
    plt.xlabel("Number of Episode")
    plt.ylabel("Mean Score");
    plt.title("Evaluation Scores Every 25 Episodes")
    plt.savefig(f'figures/Evaluation_reward.jpg')

    # Plot return of each episode
    plt.figure(2)
    plt.plot(range(0, env_config['n_episodes']+1, 1), reward_epi)
    plt.xlabel("Number of Episode")
    plt.ylabel("Score");
    plt.title("Scores of Every Episode")
    plt.savefig(f'figures/Every_reward.jpg')
