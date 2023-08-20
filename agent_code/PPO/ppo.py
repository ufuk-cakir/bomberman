import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm 
import numpy as np
import random

from collections import namedtuple, deque
#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1

K_epoch       = 3
T_horizon     = 20

hyperparameters = namedtuple('hyperparameters', ['learning_rate', 'gamma', 'lmbda', 'eps_clip', 'K_epoch', 
                                              'T_horizon', "N_FEATURES", "N_ACTIONS", "HIDDEN_SIZE", "N_LAYERS",
                                              "ANNEAL_LR","GAE",
                                              "WANDB","NUM_STEPS", "BATCH_SIZE","MINI_BATCH_SIZE",
                                              "CLIP_COEFF", "NORMALIZE_ADVANTAGE", "CLIP_VALUE_LOSS", "ENTROPY_LOSS_COEFF",
                                              "VALUE_LOSS_COEFF","MAX_GRAD_NORM","TARGET_KL","ACTIVATION_FUNCTION"])

Transition = namedtuple('Transition',
                        ('state', 'action', 'values', 'reward', 'logprob',"entropy", 'done'))


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


HYPER = hyperparameters(
    learning_rate=0.0005,
    gamma=0.98,
    lmbda=0.95,
    eps_clip=0.1,
    K_epoch=4,
    T_horizon=20,
    N_FEATURES=360,
    N_ACTIONS=len(ACTIONS),
    HIDDEN_SIZE=64,
    N_LAYERS=1,
    ANNEAL_LR=True,
    GAE=True,
    WANDB=True,
    NUM_STEPS=128,
    BATCH_SIZE=512,
    MINI_BATCH_SIZE=256,
    CLIP_COEFF=0.2,
    NORMALIZE_ADVANTAGE=True,
    CLIP_VALUE_LOSS=True,
    ENTROPY_LOSS_COEFF=0.09,#0.01,
    VALUE_LOSS_COEFF=0.4,
    MAX_GRAD_NORM=0.5,
    TARGET_KL=0.01,
    ACTIVATION_FUNCTION=nn.Tanh(),
)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Memory(object):
    '''Memory class for storing transitions
    
    This class is used to store transitions that the agent observes in a deque.
    '''
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)
    
    def push(self, *args):
        self.data.append(Transition(*args))
        
    def sample(self, batch_size):
        '''Randomly sample a batch of transitions from memory
        '''
        return random.sample(self.data, batch_size)
    
    def __len__(self):
        return len(self.data)


class PPO2(nn.Module):
    def __init__(self, HYPER):
        super(PPO2, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(HYPER.N_FEATURES,HYPER.HIDDEN_SIZE)),
            HYPER.ACTIVATION_FUNCTION,
            layer_init(nn.Linear(HYPER.HIDDEN_SIZE, HYPER.HIDDEN_SIZE)),
            HYPER.ACTIVATION_FUNCTION,
            layer_init(nn.Linear(HYPER.HIDDEN_SIZE, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(HYPER.N_FEATURES,HYPER.HIDDEN_SIZE)),
            HYPER.ACTIVATION_FUNCTION,
            layer_init(nn.Linear(HYPER.HIDDEN_SIZE, HYPER.HIDDEN_SIZE)),
            HYPER.ACTIVATION_FUNCTION,
            layer_init(nn.Linear(HYPER.HIDDEN_SIZE, HYPER.N_ACTIONS), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

'''
class PPO(nn.Module):
    def __init__(self, NUMBER_OF_POSSIBLE_ACTIONS, SIZE_OF_STATE_VECTOR):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(SIZE_OF_STATE_VECTOR, HYPER.HIDDEN_SIZE)
        self.hidden = nn.Linear(HYPER.HIDDEN_SIZE, HYPER.HIDDEN_SIZE)
        self.fc_pi = nn.Linear(HYPER.HIDDEN_SIZE, NUMBER_OF_POSSIBLE_ACTIONS)
        self.fc_v  = nn.Linear(HYPER.HIDDEN_SIZE, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=HYPER.learning_rate)
        self.loss = []

    def pi(self, x, softmax_dim = 0):
        
        Approximate optimal policy π(a|s,θ) with neural network. Takes as input the state and outputs a probability distribution over actions.
        
        Args:
            x (torch.tensor): State vector
            softmax_dim (int, optional): Dimension over which softmax is applied. Defaults to 0.
            
        Returns:
            prob (torch.tensor): Probability distribution over actions
        
        x = F.relu(self.fc1(x))
        for i in range(HYPER.N_LAYERS):
            x = F.relu(self.hidden(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    #TODO old version inefficent
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
          
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = np.array(s_prime_lst)
        done_lst = np.array(done_lst)
        prob_a_lst = np.array(prob_a_lst)

        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = zip(*self.data)
        
        s = torch.tensor(np.array(s_lst), dtype=torch.float)
        a = torch.tensor(np.array(a_lst), dtype=torch.long)
        r = torch.tensor(np.array(r_lst), dtype=torch.float)
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        prob_a = torch.tensor(np.array(prob_a_lst), dtype=torch.float)
        done_mask = torch.tensor([0 if d else 1 for d in done_lst], dtype=torch.float)
        
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
 
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(HYPER.K_epoch):
            
            td_target = r + HYPER.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = HYPER.gamma * HYPER.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-HYPER.eps_clip, 1+HYPER.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())
            self.loss.append(loss.mean().item())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
'''
