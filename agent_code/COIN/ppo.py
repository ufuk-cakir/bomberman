import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm 
import numpy as np


#Hyperparameters
class HYPER:
    SIZE_OF_STATE_VECTOR = 48
    learning_rate = 0.0005
    gamma         = 0.98
    lmbda         = 0.95
    eps_clip      = 0.1
    EPS_START     = 0.9
    EPS_END       = 0.08
    EPS_DECAY     = 3000
    N_EPOCH       = 4
    UPDATE_INTERVAL= 30
    HIDDEN_SIZE = 64
    HIDDEN_LAYER = 3#6
    ACTIVATION_FUNCTION = nn.Tanh() #ReLu, LeakyReLu, 
    MODEL_NAME = "coin_collector_new2.pt"
    


class PPO(nn.Module):
    def __init__(self, NUMBER_OF_POSSIBLE_ACTIONS, SIZE_OF_STATE_VECTOR):
        super(PPO, self).__init__()
        self.data = []
        self.fc1   = nn.Linear(SIZE_OF_STATE_VECTOR,HYPER.HIDDEN_SIZE)
        self.fc_pi = nn.Linear(HYPER.HIDDEN_SIZE,NUMBER_OF_POSSIBLE_ACTIONS)
        self.hidden = nn.Linear(HYPER.HIDDEN_SIZE,HYPER.HIDDEN_SIZE)
        self.fc_v  = nn.Linear(HYPER.HIDDEN_SIZE,1)
        self.optimizer = optim.Adam(self.parameters(), lr=HYPER.learning_rate)
        self.loss_history = []

    def pi(self, x, softmax_dim = 0):
        '''Policy Network
        
        Approximate optimal policy π(a|s,θ) with neural network. Takes as input the state and outputs a probability distribution over actions.
        
        Args:
            x (torch.tensor): State vector
            softmax_dim (int, optional): Dimension over which softmax is applied. Defaults to 0.
            
        Returns:
            prob (torch.tensor): Probability distribution over actions
        '''

        x = HYPER.ACTIVATION_FUNCTION(self.fc1(x))
        for i in range(HYPER.HIDDEN_LAYER):
            x = HYPER.ACTIVATION_FUNCTION(self.hidden(x))
        x = self.fc_pi(x)
        p = F.softmax(x, dim=softmax_dim)
        return p
    
    def v(self, x):

        x = HYPER.ACTIVATION_FUNCTION(self.fc1(x))
        for i in range(HYPER.HIDDEN_LAYER):
            x = HYPER.ACTIVATION_FUNCTION(self.hidden(x))
        v = self.fc_v(x)
        return v
      
