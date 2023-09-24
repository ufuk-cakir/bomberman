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


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



Hyperparameters = namedtuple("Hyperparameters", [
    "BATCH_SIZE",
    "GAMMA",
    "EPS_START",
    "EPS_END",
    "EPS_DECAY",
    "TAU",
    "LR",
    "N_ACTIONS",
    "N_FEATURES",
    "MLP_HIDDEN_SIZE",
    "MLP_NUM_LAYERS",
])





SIZE_OF_STATE_VECTOR = 23

HYPER = Hyperparameters(
    BATCH_SIZE=8,#8,
    GAMMA=0.99,
    EPS_START=0.80,#0.95,
    EPS_END=0.05,#0.1,
    EPS_DECAY= 3000, #1000,
    TAU=0.02,#0.01
    LR=1e-5,#5e-6
    N_ACTIONS=len(ACTIONS),
    N_FEATURES= SIZE_OF_STATE_VECTOR,
    MLP_HIDDEN_SIZE=128,
    MLP_NUM_LAYERS=2,)



import wandb


# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


class Memory(object):
    '''Memory class for storing transitions
    
    This class is used to store transitions that the agent observes in a deque.
    '''
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        '''Randomly sample a batch of transitions from memory
        '''
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)



class QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Make MLP with HYPER.MLP_NUM_LAYERS layers
        self.i2h = nn.Linear(input_size, HYPER.MLP_HIDDEN_SIZE)
         
        self.hidden = nn.ModuleList()
        for i in range(HYPER.MLP_NUM_LAYERS - 1):
            self.hidden.append(nn.Linear(HYPER.MLP_HIDDEN_SIZE, HYPER.MLP_HIDDEN_SIZE))

        self.h2o = nn.Linear(HYPER.MLP_HIDDEN_SIZE, output_size)
        
        
    def forward(self, x):
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
    
    
