# -*- coding: utf-8 -*-

# TEAM MEMBERS

# Alejandro Jarabo Peñas
# 19980430-T472
# aljp@kth.se

# Xavier de Gibert Duart
# 19970105-T477
# xdgd@kth.se

# Load packages
import dqn
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize the DISCRETE Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Initialize the Neural Network used to estimate state-value function
input_size = env.observation_space.shape[0] # 8D State Space
hidden_size_1 = 64
hidden_size_2 = 64
output_size = env.action_space.n # 4 possible actions
nn = dqn.NeuralNetwork(input_size, hidden_size_1, hidden_size_2, output_size)
nn_target = dqn.NeuralNetwork(input_size, hidden_size_1, hidden_size_2, output_size)
nn.show()

# Initialize the experience replay buffer
L = 2**14 # 2^14
C = 0 # C = 0 for generally used value
N = 2**6 # 2^6
combined = True
B = dqn.ExpRepBuffer(L, C, N, combined)
B.show()

# Train the Neural Network using DQN algorithm
gamma = 0.99
alpha = 0.0005
epsilon = 0.99
n_episodes = 1000
max_iters = 1000
eps_dec_type = 'exp'
rew_stop_th = 200
debug = False
episodes_reward, episodes_steps, episodes_epsilons = dqn.dqn(env, nn, nn_target, B, gamma, alpha, epsilon, n_episodes, max_iters, eps_dec_type, rew_stop_th, debug)

# Save trained neural network
torch.save(nn,'models/DQN_lr{}_gamma{}_neps{}_{}_NN_{}_{}_ERB_L{}_C{}_N{}_{}.pt'.format(alpha,gamma,n_episodes,eps_dec_type,hidden_size_1,hidden_size_2,L,B.C,N,'combined' if combined else ''))

# Plot Rewards and steps
n_ep_running_average = 50
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, len(episodes_reward)+1)], episodes_reward, label='Episode reward')
ax[0].plot([i for i in range(1, len(episodes_reward)+1)], running_average(episodes_reward, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward and Epsilon vs Episodes')
ax2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Epsilon')
ax2.plot([i for i in range(1, len(episodes_epsilons)+1)], episodes_epsilons, label='Episode epsilon',color='red')
ax[0].legend()
ax[0].grid(alpha=0.3)
ax[1].plot([i for i in range(1, len(episodes_steps)+1)], episodes_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, len(episodes_steps)+1)], running_average(episodes_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
plt.savefig('figs/DQN_lr{}_gamma{}_neps{}_{}_NN_{}_{}_ERB_L{}_C{}_N{}_{}.png'.format(alpha,gamma,n_episodes,eps_dec_type,hidden_size_1,hidden_size_2,L,B.C,N,'combined' if combined else ''))
