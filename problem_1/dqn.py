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
from torch import nn
from tqdm import trange
import random
import time

# NEURAL NETWORK CLASS
class NeuralNetwork(nn.Module) :
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size) :
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size)
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

    def show(self) :
        print('--------------------------------------------------\n')
        print('Neural Network Description')
        print('Input Layer Neurons = ',self.input_size)
        print('Hidden Layer 1 Neurons = ',self.hidden_size_1)
        print('Hidden Layer 2 Neurons = ',self.hidden_size_2)
        print('Output Layer Neurons = ',self.output_size,'\n')
        print(self)
        print('\n--------------------------------------------------\n')


# EXPERIENCE REPLAY BUFFER CLASS
class ExpRepBuffer :
    def __init__(self, L = 15000, C = 0, N = 64, combined=False) :
        # Experience buffer properties
        self.L = L 
        self.C = int(L/N) if C == 0 else C
        self.N = N
        # Is it a combined exp. replay buffer?
        self.combined = combined
        # Experience buffer
        self.buffer = []
    
    def init_buffer(self, env) :
        # Fill the buffer with random experiences 
        while True :
            # Initialize for new episode
            s = env.reset()
            d = False
            while not d :
                # Take random action
                a = np.random.randint(0, env.action_space.n)
                # We take one step in the environment
                next_s, r, d, _ = env.step(a)
                # Create experience
                z = (s,a,r,next_s,d)
                # Append experience to buffer
                self.add_exp(z)
                # If the episode is finished, we leave the for loop
                if d :
                    break
                # Update next state
                s = next_s
            
            # Stop when buffer is already full
            if len(self.buffer) >= self.L :
                break

    def add_exp(self,z) :
        # If buffer already full remove oldest element
        if len(self.buffer) >= self.L :
            self.buffer.pop(0)
        # Append new experience
        self.buffer.append(z)
    
    def random_batch(self) :
        if self.combined :
            batch = zip(*(random.sample(self.buffer[:-1],self.N-1)+[self.buffer[-1]]))
        else : 
            batch = zip(*(random.sample(self.buffer,self.N)))
        states, actions, rewards, next_states, dones = batch
        actions = [[actions[i]] for i in range(len(actions))]
        states_tensor = torch.tensor(np.array(states), requires_grad=False, dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(actions), requires_grad=False, dtype=torch.int64)
        rewards_tensor = torch.tensor(np.array(rewards), requires_grad=False, dtype=torch.float32)
        next_states_tensor = torch.tensor(np.array(next_states), requires_grad=False, dtype=torch.float32)
        dones_tensor = torch.tensor(np.array(dones), requires_grad=False, dtype=torch.float32)
        tensors_batch = (states_tensor,actions_tensor,rewards_tensor,next_states_tensor,dones_tensor) 
        return tensors_batch
    
    def show(self) :
        print('--------------------------------------------------\n')
        print('Experience Replay Buffer Description')
        print('Buffer size L = ',self.L)
        print('Target Network Update Period C = ',self.C)
        print('Trainig batch size N = ',self.N)
        print('Combined = ',self.combined)
        print('\n--------------------------------------------------\n')

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

def dqn(env, nn, nn_target, B, gamma=0.99, alpha = 0.0005, epsilon = 0.99, n_episodes = 500, max_iters = 1000, eps_dec_type = 'linear', rew_stop_th = 50, debug = False) :
    ''' Finds solution using eligibility SARSA
        :input Gym env            : Environment for which we want to find the best policy
        :input NeuralNetwork nn   : Neural Network whose params we want to train
        :input ExpBuffer B        : Experience replay buffer
        :input float gamma        : The discount factor.
        :input float alpha        : The learning rate.
        :input float epsilon      : The initial exploring probability.
        :input int n_episodes     : The # of episodes to simulate.
        :input int max_iters      : The max. # of steps of each episode.
        :input int rew_stop_th    : The reward threshold to stop the training.
        :input str eps_decay_type : The type of decay for epsilon.
    '''

    if debug :
        print('\nTraining DQN')
        print('Hyperparameters: ')
        print('gamma = ',gamma)
        print('alpha = ',alpha)
        print('epsilon = ',epsilon)
        print('n_episodes = ',n_episodes)
        print('max_iters = ',max_iters)
        print('eps_dec_type = ',eps_dec_type)
        print('rew_stop_th = ',rew_stop_th)
        print('debug = ',debug,'\n')

    # Available types of decay for epsilon
    eps_dec_types = ['linear','exp','step']

    # Epsilon bounds
    epsilon_max = min(0.99,epsilon)
    epsilon_min = 0.05

    # Reward collected in each episode
    n_ep_running_average = 50
    episodes_reward = []
    episodes_steps = [] 
    episodes_epsilons = []

    # Initialize target network
    nn_target.load_state_dict(nn.state_dict())

    # Initialize optimizer
    optim = torch.optim.Adam(nn.parameters(), lr=alpha)

    # Fill buffer with random experiences
    B.init_buffer(env)

    # Initialize overall number of steps
    total_steps = 0

    # trange - nice progression bar with useful information
    EPISODES = trange(n_episodes, desc='Episode: ', leave=True)

    # Iteration over episodes
    for e in EPISODES:
        # Reset enviroment
        s = env.reset()
        episode_reward = 0

        # For each step in the episode
        for i in range(max_iters):
            # Take epsilon-greedy action
            a = choose_action(env,nn,s,epsilon)
            # We take one step in the environment
            next_s, r, d, _ = env.step(a)
            # Create experience
            z = (s,a,r,next_s,d)
            # Append experience to buffer
            B.add_exp(z)
                
            # Sample batch of N experiences from B
            states, actions, rewards, next_states, dones = B.random_batch()
            
            # Set gradients to 0
            optim.zero_grad()
            
            # State-action values of next states next_s_i
            target_state_action_values_next_states = nn_target.forward(next_states)
            # Target values next states
            target_values_next_states = target_state_action_values_next_states.max(1)[0]
            # Target values 
            ys = rewards + (1-dones)*gamma*target_values_next_states
            ys = ys.clone().detach().requires_grad_(True)
            
            # Input tensor and state action values
            state_actions_values = nn.forward(states)
            # State-Action Value 
            state_action_values = state_actions_values.gather(1,actions).squeeze()

            # Update params. by performing a backward pass SGD on the MSE loss
            loss = torch.nn.functional.mse_loss(state_action_values,ys)
            # Compute gradient
            loss.backward()
            # Clip the gradient (avoid exploding gradient phenomenon)
            torch.nn.utils.clip_grad_norm_(nn.parameters(), 1)
            # Perform backpropagation
            optim.step()

            # If C steps have passed set target network equal to main network
            if total_steps == B.C : 
                total_steps = 0
                nn_target.load_state_dict(nn.state_dict())

            # Update episode rewards 
            episode_reward += r

            # If the episode is finished, we leave the for loop
            if d :
                break

            # Update next state and steps count
            s = next_s
            total_steps += 1

        # Append total episode collected reward and learning rate
        episodes_reward.append(episode_reward)
        episodes_steps.append(i)
        episodes_epsilons.append(epsilon)

        # Decay epsilon
        Z = 0.9*n_episodes
        if eps_dec_type == eps_dec_types[0] :
            epsilon = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min)*(e-1)/(Z-1))
        elif eps_dec_type == eps_dec_types[1] :
            epsilon = max(epsilon_min, epsilon_max*(epsilon_min/epsilon_max)**((e-1)/(Z-1)))
        elif eps_dec_type == eps_dec_types[2] :
            if e == int(n_episodes*(25/100)) or e == int(n_episodes*(50/100)) \
            or e == int(n_episodes*(65/100)) or e == int(n_episodes*(80/100)) :
                epsilon = epsilon/2

        # Updates the tqdm update bar with fresh information
        EPISODES.set_description(
            "Episode {} [{:.5f}] - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            e, epsilon, episode_reward, i,
            running_average(episodes_reward, n_ep_running_average)[-1],
            running_average(episodes_steps, n_ep_running_average)[-1]))

        # Stop if average reward is good enough
        if running_average(episodes_reward,n_ep_running_average)[-1] > rew_stop_th:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e, running_average(episodes_reward,n_ep_running_average)[-1]))
            break

    return episodes_reward, episodes_steps, episodes_epsilons