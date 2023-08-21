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
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

import wandb

from .ppo import HYPER

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.score = 0
    wandb.init(project="bomberman", name="ppo-OLD")
    self.global_step = 0
    self.loss_history = []
    self.reward_history = []
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=HYPER.learning_rate)
    
def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.model.make_batch()
        if len(s_prime) ==0:
            self.logger.info(f'No data to train on')
            return
        self.logger.info(f'Training on {len(s_prime)} samples')
        for i in range(HYPER.N_EPOCH):
            
            td_target = r + HYPER.gamma * self.model.v(s_prime) * done_mask
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
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-HYPER.eps_clip, 1+HYPER.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.model.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.loss_history.append(loss.mean().item())
            self.optimizer.step()  
            wandb.log({"loss": loss.mean().item()}) 


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
    self.global_step += 1
    feature_state_old = state_to_features(self,old_game_state)
    feature_state_new = state_to_features(self,new_game_state)
    
    
    
    if e.GOT_KILLED in events:
        done = True
    else:
        done = False
    action = ACTIONS.index(self_action)
    prob_a = self.prob_a
    reward = reward_from_events(self,events)
        
    self.model.put_data((feature_state_old,action,reward/100.0,feature_state_new,prob_a,done))
    self.score += reward
    self.reward_history.append(reward)
    self.logger.info(f'Score: {self.score}')
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # Idea: Add your own events to hand out rewards
    #if ...:
     #   events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    wandb.log({"reward": reward})
    if not done:
        if self.global_step % 1== 0:
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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    self.logger.info(f'Starting to train...')
    train_net(self)
    wandb.log({"mean_loss": np.mean(self.loss_history), "mean_reward": np.mean(self.reward_history),
               "cumulative_reward": self.score})
    self.logger.info(f'Training finished')
    
    # Reset the model
    self.loss_history = []
    self.score = 0
    self.global_step = 0
    
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


from .reward_shaping import custom_rewards

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    # Count number of invalid actions
    invalid_actions = events.count(e.INVALID_ACTION)
    wandb.log({"invalid_actions": invalid_actions})
    game_rewards = {
            e.COIN_COLLECTED: 1,
            e.KILLED_OPPONENT: 5,
            e.KILLED_SELF: -.6,  # idea: the custom event is bad
            e.INVALID_ACTION: -1,
            e.MOVED_DOWN:0.1,
            e.MOVED_LEFT:0.1,
            e.MOVED_RIGHT:0.1,
            e.MOVED_UP:0.1,
            e.CRATE_DESTROYED:0.1,
            e.COIN_FOUND:0.1,
            e.SURVIVED_ROUND:3,
            e.WAITED:-1,
        }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
    #reward_sum =  custom_rewards(self,events)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
