from collections import namedtuple, deque, Counter
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
                        ('state', 'action', "reward",'next_state',"prob_a","done"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

import wandb

from .ppo import HYPER

WANDB_NAME = "COIN_COLLECTOR_PPO"
WANDB_FLAG = 1

import settings

class Values:
    ''' Values to keep track of each game and reset after each game'''
    
    def __init__(self, logger=None):
        self.loss_history = []
        self.reward_history = []
        self.score = 0
        self.global_step = 0
        self.invalid_actions = 0
        self.waited_for = 0
        self.event_history = []
        self.data = []
        self.logger = logger
        
    def reset(self):
        self.loss_history = []
        self.reward_history = []
        self.event_history = []
        self.score = 0
        self.global_step = 0
        self.invalid_actions = 0
        self.waited_for = 0
        self.data = []
        self.invalid_actions = 0

        
    def add_loss(self,loss,):
        self.loss_history.append(loss)
        wandb.log({"loss_step": loss}) if WANDB_FLAG else None
    
    def add_reward(self,reward):
        self.reward_history.append(reward)
        self.score += reward
        wandb.log({"reward_step": reward}) if WANDB_FLAG else None

    def add_invalid_action(self):
        self.invalid_actions += 1
    def waited(self):
        self.waited_for += 1
       
    def step(self):
        self.global_step += 1 
    
    def log_wandb_end(self):
        wandb.log({"mean_loss": np.mean(self.loss_history), "mean_reward": np.mean(self.reward_history),
               "cumulative_reward": self.score,
               "invalid_actions_per_game": self.invalid_actions}) if WANDB_FLAG else None
        self.logger.info(f"END OF GAME: mean_loss: {np.mean(self.loss_history)}, cumulative_reward: {self.score}")
        
        
        self.logger.info(f'Event stats:')
        # Convert list of tuples to list of strings
        event_strings = [item for tup in self.event_history for item in tup]

        # Now you can use Counter on this list of strings
        event_counts = Counter(event_strings)
        for event, count in event_counts.items():
            self.logger.info(f'{event}: {count}')
            if WANDB_FLAG:
                wandb.log({f"event_{event}": count})

        self.logger.info("--------------------------------------")
        
        self.reset()
        wandb.save(HYPER.MODEL_NAME) if WANDB_FLAG else None
        #wandb.save("logs/PPOLD.log") if WANDB_FLAG else None
    
    def add_event(self,event):
        self.event_history.append(event)
        
    def check_repetition(self):
        # Check if agent is repeating actions, i.e going back and forth
        if self.global_step > 2:
            # Check specfically for left-right-left-right
            if "MOVED_LEFT" in self.event_history[-2:] and "MOVED_RIGHT" in self.event_history[-4:-2]:
                self.add_event("REPEATING_ACTIONS")
                self.logger.info(f'Agent is repeating actions')
                return True
            if "MOVED_RIGHT" in self.event_history[-2:] and "MOVED_LEFT" in self.event_history[-4:-2]:
                self.add_event("REPEATING_ACTIONS")
                self.logger.info(f'Agent is repeating actions')
                return True
            if "MOVED_UP" in self.event_history[-2:] and "MOVED_DOWN" in self.event_history[-4:-2]:
                self.add_event("REPEATING_ACTIONS")
                self.logger.info(f'Agent is repeating actions')
                return True
            if "MOVED_DOWN" in self.event_history[-2:] and "MOVED_UP" in self.event_history[-4:-2]:
                self.add_event("REPEATING_ACTIONS")
                self.logger.info(f'Agent is repeating actions')
                return True
            
        
    def push_data(self, transition):
        # Add reward to reward history
        transition = Transition(*transition)
        self.add_reward(transition.reward)
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
        
    




def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
 
    self.score = 0
    if WANDB_FLAG:
        wandb.init(project="bomberman", name=WANDB_NAME)
        for key, value in HYPER.__dict__.items():
            if not key.startswith("__"):
                wandb.config[key] = value
    #self.global_step = 0

    #self.loss_history = []
    #self.reward_history = []
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=HYPER.learning_rate)
    self.invalid_actions = 0
    
    self.values = Values(   logger=self.logger)

   
          
def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.values.make_batch()
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
            
            #self.loss_history.append(loss.mean().item())
            self.optimizer.step()  
            self.values.add_loss(loss.mean().item())
        

#----------TODO change this to custom one
#from .reward_shaping import custom_rewards, reward_coin_distance, check_placed_bomb, check_blast_radius, CLOSER_TO_COIN, FURTHER_FROM_COIN, ESCAPABLE_BOMB, BOMB_DESTROYS_CRATE, WAITED_TOO_LONG, IN_BLAST_RADIUS
WAITED_TOO_LONG = "WAITED_TOO_LONG"
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
    self.values.step()
    
    feature_state_old = state_to_features(self,old_game_state)
    feature_state_new = state_to_features(self,new_game_state)
    
    
    
    if e.GOT_KILLED in events:
        done = True
    else:
        done = False
    action = ACTIONS.index(self_action)
    prob_a = self.prob_a
    
    # Custom Rewards
    #reward_coin_distance(old_game_state, new_game_state, events)
    #punish_long_wait(self,events)
    max_wait = settings.EXPLOSION_TIMER
    if e.WAITED in events:
        self.values.waited_for += 1
    else:
        self.values.waited_for = 0
    if self.values.waited_for > max_wait:
        events.append(WAITED_TOO_LONG)
    
    
    if e.INVALID_ACTION in events:
        self.values.add_invalid_action()
    # TODO: FIX THIS
    #check_placed_bomb(feature_state_old, new_game_state, events)
    #check_blast_radius(old_game_state,events)
    
    
    check_custom_events(self,events,action, feature_state_old, feature_state_new)
    
    self.values.add_event(events)
    reward = reward_from_events(self,events)
        
    
    
    
    
    self.values.push_data((feature_state_old,action,reward/100.0,feature_state_new,prob_a,done))
    
    #self.values.add_reward(reward)
    
    #self.score += reward
    #self.reward_history.append(reward)
    #self.logger.info(f'Score: {self.values.score}')
    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # Idea: Add your own events to hand out rewards
    #if ...:
     #   events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
   
   
    
    
    if not done:
        if self.values.global_step % 128== 0:
            
            train_net(self)
            self.values.log_wandb_end()
            
           # self.loss_history = []
            

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
    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    #self.logger.info(f'Starting to train after end...')
    train_net(self)
    
    # Log and reset values
    self.values.log_wandb_end()
    
    # Reset values defined in setup
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # Store the model
    with open(HYPER.MODEL_NAME, "wb") as file:
        pickle.dump(self.model, file)



REPEATING_ACTIONS = "REPEATING_ACTIONS"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"
ESCAPABLE_BOMB = "ESCAPABLE_BOMB"
BOMB_DESTROYS_CRATE = "BOMB_DESTROYS_CRATE"

TOOK_DIRECTION_TOWARDS_TARGET = "TOOK_DIRECTION_TOWARDS_TARGET"
TOOK_DIRECTION_AWAY_FROM_TARGET = "TOOK_DIRECTION_AWAY_FROM_TARGET"

IS_IN_LOOP = "IS_IN_LOOP"
GOT_OUT_OF_LOOP = "GOT_OUT_OF_LOOP"


        
def check_custom_events(self, events: List[str],action, features_old, features_new):
    # Check if agent is closer to coin
    
    
    nearest_coin_distance_old, is_dead_end_old, bomb_threat_old, time_to_explode_old,\
        can_drop_bomb_old, is_next_to_opponent_old, is_on_bomb_old, crates_nearby_old,\
            escape_route_available_old, direction_to_target_old_UP, direction_to_target_old_DOWN,\
                direction_to_target_old_LEFT, direction_to_target_old_RIGHT, is_in_loop_old = features_old
    
    nearest_coin_distance_new, is_dead_end_new, bomb_threat_new, time_to_explode_new,\
        can_drop_bomb_new, is_next_to_opponent_new, is_on_bomb_new, crates_nearby_new,\
            escape_route_available_new, direction_to_target_new_UP, direction_to_target_new_DOWN,\
                direction_to_target_new_LEFT, direction_to_target_new_RIGHT, is_in_loop_new = features_new
                
                
    
    
    # Feature list: [nearest_coin_distance, is_dead_end, bomb_threat, time_to_explode, can_drop_bomb, 
    # is_next_to_opponent, is_on_bomb, crates_nearby, escape_route_available, direction_to_target, is_in_loop, ignore_others_timer_normalized]
    if nearest_coin_distance_new < nearest_coin_distance_old:
        events.append(CLOSER_TO_COIN)
    else: 
        events.append(FURTHER_FROM_COIN)
        
    # Check if bomb is escapable
    if is_on_bomb_new and not is_on_bomb_old:
        events.append(ESCAPABLE_BOMB)
    # Check if bomb destroys crate
 
    # Check if Agent took direction towards target: direction_to_target = [UP, DOWN, LEFT, RIGHT]
    if (action == "MOVED_UP" and direction_to_target_old_UP == 1) or (action == "MOVED_DOWN" and direction_to_target_old_DOWN == 1)\
        or (action == "MOVED_LEFT" and direction_to_target_old_LEFT == 1) or (action == "MOVED_RIGHT" and direction_to_target_old_RIGHT == 1):
        events.append(TOOK_DIRECTION_TOWARDS_TARGET)
    else:
        events.append(TOOK_DIRECTION_AWAY_FROM_TARGET) # TODO: check if this is correct
      
    # TODO: check if this works, i think its useless atm  
    if is_in_loop_new:
        events.append(IS_IN_LOOP)
    if not is_in_loop_new and is_in_loop_old:
        events.append(GOT_OUT_OF_LOOP)
    

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    # Count number of invalid actions
    if e.INVALID_ACTION in events:
        self.invalid_actions += 1
    """ 
    game_rewards = {
        e.COIN_COLLECTED: 300,
        e.KILLED_OPPONENT: 700,
        #e.OPPONENT_ELIMINATED: 700,
        e.KILLED_SELF: -400,
        # e.BOMB_DROPPED: 10,
        e.COIN_FOUND: 10,
        e.CRATE_DESTROYED: 10,
        e.INVALID_ACTION: -50,
        # e.SURVIVED_ROUND: 100,
        BOMB_DESTROYS_CRATE: 40,
        ESCAPABLE_BOMB: 75,
        e.MOVED_LEFT: -10,
        e.MOVED_DOWN: -10,
        e.MOVED_RIGHT: -10,
        e.MOVED_UP: -10,
        e.WAITED: -50,
        WAITED_TOO_LONG: -350,
        CLOSER_TO_COIN: 40,
        IN_BLAST_RADIUS: -7,
        # e.BOMB_DROPPED: -1,
        FURTHER_FROM_COIN: -100,
        CLOSER_TO_COIN:70,
        
    }
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -8,  
        e.INVALID_ACTION: -15,
        #e.MOVED_DOWN:0.5,
        #e.MOVED_LEFT:0.5,
        #e.MOVED_RIGHT:0.5,
        #e.MOVED_UP:0.5,
        e.CRATE_DESTROYED:1,
        e.COIN_FOUND:0.1,
        e.SURVIVED_ROUND:3,
    }
    custom = {
        WAITED_TOO_LONG:-5,
        IN_BLAST_RADIUS:-1,
        FURTHER_FROM_COIN:-10,
        CLOSER_TO_COIN: 7,
        ESCAPABLE_BOMB: 1,
        REPEATING_ACTIONS:-25,
    }
    # TODO add punishment for visiting same fields
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        if event in custom:
            reward_sum += custom[event]
        #    self.logger.info(f'Awarded {custom[event]} for event {event}')
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    # Useless bomb
    if self.values.check_repetition():
        reward_sum += custom[REPEATING_ACTIONS]
    if e.BOMB_DROPPED in events and BOMB_DESTROYS_CRATE not in events:
        reward_sum -= 10
        #self.logger.info(f'Useless bomb: -10')
        # print("useless")
    if e.BOMB_DROPPED in events and BOMB_DESTROYS_CRATE in events:
        reward_sum += 15
        #self.logger.info(f'Useful bomb, bomb destroyed crate: +15')
    # Inescapable bomb
    if e.BOMB_DROPPED in events and ESCAPABLE_BOMB not in events:
        reward_sum += game_rewards[e.KILLED_SELF]
        #self.logger.info(f'Inescapable bomb: {game_rewards[e.KILLED_SELF]}')
    # Reward for going out of blast radius
    # TODO: check if this works    
    
    
    
"""

    
    

    #--------------_FOR COIN COLLECTOR AGENT
    coin_rewards = {
        e.COIN_COLLECTED: 20,
        FURTHER_FROM_COIN:-10,
        CLOSER_TO_COIN: 7,
        e.BOMB_DROPPED:-20,
        e.INVALID_ACTION:-10,
    }
    reward_sum = 0
    for event in events:
        if event in coin_rewards:
            reward_sum += coin_rewards[event]
        
    #self.reward_history.append(reward_sum)
    return reward_sum
    #reward_sum =  custom_rewards(self,events)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
