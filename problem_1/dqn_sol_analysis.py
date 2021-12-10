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
import plotly.graph_objects as go

# HELPER FUNCTIONS
def plot_opt_policy_or_value_func (nn,type) :
    # Sample Qw(s,a) values
    y_low, y_high = 0, 1.5
    beta_low, beta_high = -np.pi, np.pi
    y = np.linspace(y_low, y_high, 100)
    beta = np.linspace(beta_low, beta_high, 100)

    # Sample values from continuous function
    V = np.zeros((100,100))
    for k1 in range(len(y)) :
        y_i = y[k1]
        for k2 in range(len(beta)):
            beta_i = beta[k2]
            # Create input tensor
            state = [0,y_i,0,0,beta_i,0,0,0]
            input_tensor =  torch.tensor(np.array(state), requires_grad=False, dtype=torch.float32)
            state_action_values = nn.forward(input_tensor)
            if type == 'opt_policy' :
                V[(k1,k2)] = state_action_values.max(0)[1].item()
            elif type == 'value_func' : 
                V[(k1,k2)] = state_action_values.max(0)[0].item()

    fig = go.Figure(data =
        go.Contour(
            z=V,
            x=y, # horizontal axis
            y=beta, # vertical axis
            colorbar=dict(
                title='Optimal action' if type == 'opt_policy' else 'State Value V(s)',
            )
        ))
    fig.update_layout(
        title='Best Policy' if type == 'opt_policy' else 'Value Function',
        xaxis_title='Height '+'$y$',
        yaxis_title='Angle '+'$\\beta$',
    )
    fig.write_image('figs/'+('best_policy' if type == 'opt_policy' else 'value_function')+'.png')

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

## Plot value function and optimal policy depending on params
plot_opt_policy_or_value_func(nn,'opt_policy')
plot_opt_policy_or_value_func(nn,'value_func')
