from collections import namedtuple, deque
import numpy as np
import pickle
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

import events as e
from .callbacks import state_to_features, ACTIONS

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', "done","prob_a"))

# Hyper parameters -- DO modify
 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

import wandb

from .ppo import HYPER

import random

WANDB_FLAG = True
class Memory:
    def __init__(self):
        self.memory = deque(maxlen=HYPER.TRANSITION_HISTORY_SIZE)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = Memory()
    self.score = 0
    wandb.init(project="bomberman", name="ppo-OLD")if WANDB_FLAG else None
    self.global_step = 0
    self.round_step = 0
    self.loss_history = []
    self.reward_history = []
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=HYPER.learning_rate)


def sample_batches(self,size):
    samples_transitions = self.transitions.sample(size)
    batch = Transition(*zip(*samples_transitions))
    states = np.array(batch.state)
    state_batch = torch.tensor(states, dtype=torch.float)
    action_batch = torch.tensor(np.array(batch.action))
    reward_batch = torch.tensor(np.array(batch.reward))
    next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float)
    done_batch = torch.tensor(np.array(batch.done), dtype=torch.float)
    prob_a_batch = torch.tensor(np.array(batch.prob_a))

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch, prob_a_batch


def train_net(self):
        # Sample from memory buffer
        # Check if there is enough data to train on
        if len(self.transitions) < HYPER.BATCH_SIZE:
            # Train on all data available
            size = len(self.transitions)
        else: 
            size = HYPER.BATCH_SIZE
        s,a,r,s_prime,done_mask,prob_a = sample_batches(self, size)
     
        #s, a, s_prime,r, done_mask, prob_a = self.transitions.sample(size)
        #s, a, r, s_prime, done_mask, prob_a = self.model.make_batch()
        if len(s_prime) ==0:
            self.logger.info(f'No data to train on')
            return
        self.logger.info(f'Training on {len(s_prime)} samples')
        for i in range(HYPER.N_EPOCH):
            
            td_target = r + HYPER.gamma * self.model.v(s_prime) * done_mask #Maybe a problem if s_prime is None
            delta = td_target - self.model.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = HYPER.gamma * HYPER.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.model.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a.view(-1,1))
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-HYPER.eps_clip, 1+HYPER.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.model.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.loss_history.append(loss.mean().item())
            self.optimizer.step()  
            
            
            
def calculate_and_store(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.global_step += 1
    self.round_step += 1
    feature_state_old = state_to_features(self,old_game_state)
    feature_state_new = state_to_features(self,new_game_state) if new_game_state is not None else np.zeros_like(feature_state_old)
    
    
    if e.GOT_KILLED in events:
        done = True
    elif e.KILLED_SELF in events:
        done = True
    elif feature_state_new is None:
        done = True
    else:
        done = False
    action = ACTIONS.index(self_action)
    prob_a = self.prob_a
    reward = reward_from_events(self,events)
    
    self.transitions.push(feature_state_old,action,feature_state_new,reward/100.0,done,prob_a)
    #self.model.put_data((feature_state_old,action,reward/100.0,feature_state_new,prob_a,done))
    self.score += reward
    self.reward_history.append(reward)
    self.logger.info(f'Score: {self.score}')
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {self.global_step} due to action {self_action}. Reward: {reward}')
    # Idea: Add your own events to hand out rewards
    #if ...:
     #   events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    wandb.log({"reward": reward})if WANDB_FLAG else None
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    calculate_and_store(self, old_game_state, self_action, new_game_state, events)
    done = self.transitions.memory[-1].done
    if not done:
        if self.global_step % HYPER.TRANSITION_HISTORY_SIZE == 0:
            self.logger.info(f'Starting to train after end: total steps {self.global_step}')
            train_net(self)
            self.loss_history = []
            

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    calculate_and_store(self, last_game_state, last_action, None, events)
    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    self.logger.info(f'Starting to train...')
    train_net(self)
    self.logger.info(f"---------END OF ROUND: Total Steps{self.round_step}-----------")
    wandb.log({"mean_loss": np.mean(self.loss_history), "mean_reward": np.mean(self.reward_history),
               "cumulative_reward": self.score})if WANDB_FLAG else None
   
    self.logger.info
    # Reset the model
    self.loss_history = []
    self.score = 0
    self.round_step = 0
    self.reward_history = []
    
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    wandb.save("my-saved-model.pt")if WANDB_FLAG else None
    wandb.save("logs/PPOLD.log") if WANDB_FLAG else None

from .reward_shaping import custom_rewards

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    reward_sum =  custom_rewards(self,events)
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
