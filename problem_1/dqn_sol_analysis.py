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

# load solution neural network
nn = torch.load('models/DQN_lr0.0005_gamma0.99_neps1000_exp_NN_64_64_ERB_L16384_C256_N64_combined.pt')

print(nn)