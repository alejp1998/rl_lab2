# -*- coding: utf-8 -*-

# TEAM MEMBERS

# Alejandro Jarabo Peñas
# 19980430-T472
# aljp@kth.se

# Xavier de Gibert Duart
# 19970105-T477
# xdgd@kth.se

# Load packages
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt

## Helper functions
def choose_action(env,nn, s, epsilon) :
    ''' Choose next action based on actual state '''
    # We sample a float from uniform distribution
    explore = np.random.uniform(0,1) < epsilon
    # If it's lower than epsilon, we select random action (EXPLORE)
    if explore :
        a = np.random.randint(0, env.action_space.n)
    # If it's higher we take best action we have learned so far (EXPLOIT)
    else :
        # Sample state-action values
        s_tensor = torch.tensor(s,dtype=torch.float32,requires_grad = False)
        state_action_values = nn.forward(s_tensor)
        # Choose action with higher state-action value
        a = state_action_values.max(0)[1].item()
    return a

# Import and initialize the DISCRETE Lunar Laner Environment
env_random = gym.make('LunarLander-v2')
env_dqn = gym.make('LunarLander-v2')
env_random.reset()
env_dqn.reset()

# LOAD TRAINED DQN MODEL
# Neural Network params
hidden_size_1 = 64
hidden_size_2 = 64
# Experience Replay Buffer Params
L = 2**14 # 2^14
N = 2**6 # 2^6
C = int(L/N)
combined = True
# Hyperparameters
gamma = 0.99
alpha = 0.0005
epsilon = 0.99
n_episodes = 2000
max_iters = 1000
eps_dec_type = 'exp'
# Load the model
nn = torch.load('models/DQN_lr{}_gamma{}_neps{}_{}_NN_{}_{}_ERB_L{}_C{}_N{}_{}.pt'.format(alpha,gamma,n_episodes,eps_dec_type,hidden_size_1,hidden_size_2,L,C,N,'combined' if combined else ''))
print(nn)

## SIMULATE OVER N EPISODES
N = 50
episodes_reward_random = []
episodes_reward_dqn = []
episodes_steps_random = []
episodes_steps_dqn = []
for e in range(N) :
    # Reset enviroments and episode reward
    s_random = env_random.reset()
    s_dqn = env_dqn.reset()
    episode_reward_random = 0
    episode_reward_dqn = 0
    d_random, d_dqn = False, False

    # For each step in the episode
    for i in range(max_iters):
        # If the episodes are finished, we leave the for loop
        if d_random and d_dqn :
            break
        if not d_random :
            a_random = choose_action(env_random,nn,s_random,1)
            next_s_random, r_random, d_random, _ = env_random.step(a_random)
            episode_reward_random += r_random
            s_random = next_s_random
            i_max_random = i
        if not d_dqn :
            a_dqn = choose_action(env_dqn,nn,s_dqn,0)
            next_s_dqn, r_dqn, d_dqn, _ = env_dqn.step(a_dqn)
            episode_reward_dqn += r_dqn
            s_dqn = next_s_dqn
            i_max_dqn = i
    
    episodes_reward_random.append(episode_reward_random)
    episodes_reward_dqn.append(episode_reward_dqn)
    episodes_steps_random.append(i_max_random)
    episodes_steps_dqn.append(i_max_dqn)


# PLOT RESULTS
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
ax[0,0].plot([i for i in range(1, len(episodes_reward_random)+1)], episodes_reward_random, label='Episode reward')
ax[0,0].set_xlabel('Episodes')
ax[0,0].set_ylabel('Total reward')
ax[0,0].set_title('RANDOM - Total Reward vs Episodes')
ax[0,0].legend()
ax[0,0].grid(alpha=0.3)
ax[0,1].plot([i for i in range(1, len(episodes_steps_random)+1)], episodes_steps_random, label='Steps per episode')
ax[0,1].set_xlabel('Episodes')
ax[0,1].set_ylabel('Total number of steps')
ax[0,1].set_title('RANDOM - Total number of steps vs Episodes')
ax[0,1].legend()
ax[0,1].grid(alpha=0.3)
ax[1,0].plot([i for i in range(1, len(episodes_reward_dqn)+1)], episodes_reward_dqn, label='Episode reward')
ax[1,0].set_xlabel('Episodes')
ax[1,0].set_ylabel('Total reward')
ax[1,0].set_title('DQN - Total Reward vs Episodes')
ax[1,0].legend()
ax[1,0].grid(alpha=0.3)
ax[1,1].plot([i for i in range(1, len(episodes_steps_dqn)+1)], episodes_steps_dqn, label='Steps per episode')
ax[1,1].set_xlabel('Episodes')
ax[1,1].set_ylabel('Total number of steps')
ax[1,1].set_title('DQN - Total number of steps vs Episodes')
ax[1,1].legend()
ax[1,1].grid(alpha=0.3)
plt.show()
plt.savefig('figs/random-vs-DQN_lr{}_gamma{}_neps{}_{}_NN_{}_{}_ERB_L{}_C{}_N{}_{}.png'.format(alpha,gamma,n_episodes,eps_dec_type,hidden_size_1,hidden_size_2,L,C,N,'combined' if combined else ''))

        
        

