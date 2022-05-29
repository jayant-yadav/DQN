# Reinforcement Learning DQN Project
This directory contains all code needed to train an agent to play Pong or CartPole.

The **models/permanent** directory contains models for CartPole and Pong trained during 1000 episodes, and Pong during 3000 episodes. These all use the default parameters.

In order to run the 3000 model, some lines in the **forward** function in **DQN.py** need to be commented out, see the file for details. This is due to the network being extended to also support CartPole after training this agent.