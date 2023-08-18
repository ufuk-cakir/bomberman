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
    "N_FEATURES"
])


SIZE_OF_STATE_VECTOR = 316


HYPER = Hyperparameters(
    BATCH_SIZE=128,
    GAMMA=0.99,
    EPS_START=0.9,
    EPS_END=0.05,
    EPS_DECAY=1000,
    TAU=0.005,
    LR=1e-4,
    N_ACTIONS=len(ACTIONS),
    N_FEATURES=SIZE_OF_STATE_VECTOR
)



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
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
    
