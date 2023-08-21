import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm 
import random
import math 
import numpy as np
from collections import namedtuple, deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html






SIZE_OF_STATE_VECTOR = 360


class HYPER:
    BATCH_SIZE=128,
    GAMMA=0.98,
    EPS_START=1.0,#0.9,
    EPS_END=0.1,#0.05,
    EPS_DECAY= 2000, #1000,
    TAU=0.005,
    LR=1e-3,
    N_ACTIONS=len(ACTIONS),
    N_FEATURES=SIZE_OF_STATE_VECTOR,
    MLP_HIDDEN_SIZE=128,
    MLP_NUM_LAYERS=2,
    MEMORY_SIZE= 1e6,
    K_FEATURES=4,
    MAX_EPISODE_NUM = 200000,
    
    
    
import wandb


# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward','done', 'next_state', 'goal'))


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, *item):
        self.memory.append(Transition(*item))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Make MLP with HYPER.MLP_NUM_LAYERS layers
        self.i2h = nn.Linear(input_size, HYPER.MLP_HIDDEN_SIZE)
         
        self.hidden = nn.ModuleList()
        for i in range(HYPER.MLP_NUM_LAYERS - 1):
            layer = nn.Linear(HYPER.MLP_HIDDEN_SIZE, HYPER.MLP_HIDDEN_SIZE)
            nn.init.kaiming_normal_(layer.weight)
            layer.bias.data.zero_()
            self.hidden.append(layer)

        self.h2o = nn.Linear(HYPER.MLP_HIDDEN_SIZE, output_size)
        nn.init.xavier_uniform_(self.h2o.weight)
        self.h2o.bias.data.zero_()
        
    def forward(self, states, goals):
        x = torch.cat((states, goals), dim=1)
        x = F.relu(self.i2h(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.h2o(x)
'''
    
    
class QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.i2h = nn.Linear(input_size, HYPER.MLP_HIDDEN_SIZE)
         
        self.hidden = nn.ModuleList()
        for i in range(HYPER.MLP_NUM_LAYERS - 1):
            self.hidden.append(nn.Linear(HYPER.MLP_HIDDEN_SIZE, HYPER.MLP_HIDDEN_SIZE))
            self.hidden.append(nn.BatchNorm1d(HYPER.MLP_HIDDEN_SIZE))
            self.hidden.append(nn.Dropout(0.2))  # Adding dropout

        self.h2o = nn.Linear(HYPER.MLP_HIDDEN_SIZE, output_size)
        
        # Initialize weights using Xavier initialization
        for layer in [self.i2h] + list(self.hidden) + [self.h2o]:  # Convert ModuleList to list
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
    def forward(self, x):
        x = F.relu(self.i2h(x))
        for layer in self.hidden:
            x = layer(x)
            x = F.leaky_relu(x, negative_slope=0.01)  # Leaky ReLU activation
        return self.h2o(x)


'''
    
    
