import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm 
import numpy as np



#Hyperparameters

class HYPER:
    learning_rate = 0.0005
    gamma         = 0.98
    lmbda         = 0.95
    eps_clip      = 0.1
    EPS_START     = 0.9
    EPS_END       = 0.05
    EPS_DECAY     = 1000
    N_EPOCH       = 3
    UPDATE_INTERVAL     = 30
    HIDDEN_SIZE = 64
    HIDDEN_LAYER = 1
    ACTIVATION_FUNCTION = nn.ReLU()


class PPO(nn.Module):
    def __init__(self, NUMBER_OF_POSSIBLE_ACTIONS=4, SIZE_OF_STATE_VECTOR=5):
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
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = HYPER.ACTIVATION_FUNCTION(self.fc1(x))
        for i in range(HYPER.HIDDEN_LAYER):
            x = HYPER.ACTIVATION_FUNCTION(self.hidden(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
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
        
        #convert to numpy arrays
        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = np.array(s_prime_lst)
        done_lst = np.array(done_lst)

        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    
'''

def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()

'''        