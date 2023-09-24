from collections import namedtuple, deque, Counter
import numpy as np
import pickle
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings
import wandb
import events as e
from .callbacks import state_to_features, ACTIONS, LOG_WANDB, DEBUG_EVENTS, LOG_TO_FILE
from .ppo import HYPER


Transition = namedtuple('Transition',
                        ('state', 'action', "reward",'next_state',"prob_a","done"))



# Parameters
log_to_file = LOG_TO_FILE
train_in_round = True
TRAIN_EVERY_N_STEPS = 128
TRAIN_EVERY_END_OF_ROUND = False




# Events
 
WAITED_TOO_LONG = "WAITED_TOO_LONG"
BOMB_WAS_USELESS = "BOMB_WAS_USELESS"
AGENT_CHOSE_INVALID_DIRECTION = "AGENT_CHOSE_INVALID_DIRECTION"
DROPPED_BOMB_AND_COLLECTED_COIN_MEANWHILE = "DROPPED_BOMB_AND_COLLECTED_COIN_MEANWHILE"
WENT_INTO_BOMB_RADIUS_AND_DIED = "WENT_INTO_BOMB_RADIUS_AND_DIED"
BLAST_COUNT_RIGHT_DECREASED = "BLAST_COUNT_RIGHT_DECREASED"
BLAST_COUNT_LEFT_DECREASED = "BLAST_COUNT_LEFT_DECREASED"
BLAST_COUNT_DOWN_DECREASED = "BLAST_COUNT_DOWN_DECREASED"
BLAST_COUNT_UP_DECREASED = "BLAST_COUNT_UP_DECREASED"
GOING_AWAY_FROM_BOMB = "GOING_AWAY_FROM_BOMB"
GOING_TOWARDS_BOMB = "GOING_TOWARDS_BOMB"
ESCAPED_BOMB = "ESCAPED_BOMB"
IN_BLAST_RADIUS = "IN_BLAST_RADIUS"
GOT_OUT_OF_LOOP = "GOT_OUT_OF_LOOP"
IS_IN_LOOP = "IS_IN_LOOP"
TOOK_DIRECTION_AWAY_FROM_TARGET = "TOOK_DIRECTION_AWAY_FROM_TARGET"
TOOK_DIRECTION_TOWARDS_TARGET = "TOOK_DIRECTION_TOWARDS_TARGET"
BOMB_DESTROYS_CRATE = "BOMB_DESTROYS_CRATE"
ESCAPABLE_BOMB = "ESCAPABLE_BOMB"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
REPEATING_ACTIONS = "REPEATING_ACTIONS"
WANDB_FLAG = LOG_WANDB
WANDB_NAME = "COIN_COLLECTOR_WITHOUT_DIRECTION"


    

class Values: 
    ''' Values to keep track of each game and reset after each game'''
    def __init__(self, logger=None):
        self.random_actions = 0
        self.coordinate_history = deque([], 5)
        self.loss_history = []
        self.reward_history = []
        self.score = 0
        self.global_step = 0
        self.invalid_actions = 0
        self.waited_for = 0
        self.event_history = []
        self.data = []
        self.logger = logger
        self.frames_after_bomb = 0
        
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
        self.frames_after_bomb = 0
        self.coordinate_history = deque([], 5)
        self.random_actions = 0

        
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
        self.logger.info(f"END OF GAME: mean_loss: {np.mean(self.loss_history)}, cumulative_reward: {self.score}") if log_to_file else None
        
        
        self.logger.info(f'Event stats:') if log_to_file else None
        if WANDB_FLAG:
            wandb.log({"global_step": self.global_step})
        # Convert list of tuples to list of strings
        event_strings = [item for tup in self.event_history for item in tup]

        # Now you can use Counter on this list of strings
        event_counts = Counter(event_strings)
        for event, count in event_counts.items():
            self.logger.info(f'{event}: {count}')if log_to_file else None
            if WANDB_FLAG:
                wandb.log({f"event_{event}": count})

        self.logger.info("--------------------------------------")if log_to_file else None
        
        self.reset()
        wandb.save(HYPER.MODEL_NAME) if WANDB_FLAG else None
        
    
    def add_event(self,event):
        self.event_history.append(event)
        
    def check_repetition(self):
        # Check if agent is repeating actions, i.e going back and forth
        if self.global_step > 2:
            # Check specfically for left-right-left-right
            if "MOVED_LEFT" in self.event_history[-2:] and "MOVED_RIGHT" in self.event_history[-4:-2]:
                self.add_event("REPEATING_ACTIONS")
                self.logger.info(f'Agent is repeating actions') if log_to_file else None
                return True
            if "MOVED_RIGHT" in self.event_history[-2:] and "MOVED_LEFT" in self.event_history[-4:-2]:
                self.add_event("REPEATING_ACTIONS")
                self.logger.info(f'Agent is repeating actions')if log_to_file else None
                return True
            if "MOVED_UP" in self.event_history[-2:] and "MOVED_DOWN" in self.event_history[-4:-2]:
                self.add_event("REPEATING_ACTIONS") 
                self.logger.info(f'Agent is repeating actions')
                return True
            if "MOVED_DOWN" in self.event_history[-2:] and "MOVED_UP" in self.event_history[-4:-2]:
                self.add_event("REPEATING_ACTIONS")
                self.logger.info(f'Agent is repeating actions') if log_to_file else None
                return True
            
        
    def push_data(self, transition):
        # Add reward to reward history
        transition = Transition(*transition)
        self.add_reward(transition.reward)
        self.data.append(transition)
        
    
    def make_batch(self):
        # Initialize lists to store batch data
        batch = {
            's': [],
            'a': [],
            'r': [],
            's_prime': [],
            'prob_a': [],
            'done_mask': []
        }

        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            # Append data to respective lists
            batch['s'].append(s)
            batch['a'].append([a])
            batch['r'].append([r])
            batch['s_prime'].append(s_prime)
            batch['prob_a'].append([prob_a])
            done = 0 if done else 1
            batch['done_mask'].append([done])

        # Convert lists to numpy arrays
        for key in batch.keys():
            batch[key] = np.array(batch[key])

        # Convert numpy arrays to PyTorch tensors
        s = torch.tensor(batch['s'], dtype=torch.float)
        a = torch.tensor(batch['a'])
        r = torch.tensor(batch['r'])
        s_prime = torch.tensor(batch['s_prime'], dtype=torch.float)
        done_mask = torch.tensor(batch['done_mask'], dtype=torch.float)
        prob_a = torch.tensor(batch['prob_a'])

        # Clear the data buffer
        self.data = []

        return s, a, r, s_prime, done_mask, prob_a
        
    


def train_net(self):
    s, a, r, s_prime, done_mask, prob_a = self.values.make_batch()
    

    s = s.to(self.device)
    a = a.to(self.device)
    r = r.to(self.device)
    s_prime = s_prime.to(self.device)
    done_mask = done_mask.to(self.device)
    prob_a = prob_a.to(self.device)
    
    if len(s_prime) ==0:
        self.logger.info(f'No data to train on')
        return
    self.logger.info(f'Training on {len(s_prime)} samples')
    for i in range(HYPER.N_EPOCH):
        
        td_target = r + HYPER.gamma * self.model.v(s_prime) * done_mask
        delta = td_target - self.model.v(s)
        delta = delta.detach().cpu().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = HYPER.gamma * HYPER.lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

        pi = self.model.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-HYPER.eps_clip, 1+HYPER.eps_clip) * advantage
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.model.v(s) , td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()  
        self.values.add_loss(loss.mean().item())

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.score = 0
    if WANDB_FLAG:
        wandb.init(project="bomberman", name=WANDB_NAME)
        for key, value in HYPER.__dict__.items():
            if not key.startswith("__"):
                wandb.config[key] = value
        wandb.watch(self.model)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=HYPER.learning_rate)
    self.invalid_actions = 0
    
    self.values = Values(logger=self.logger)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.logger.info(f'Using device: {self.device}')
    self.model = self.model.to(self.device)
    self.best_score = 0
   
          

        


def calculate_events_and_reward(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.values.step()
    
    feature_state_old = state_to_features(self,old_game_state)
    feature_state_new = state_to_features(self,new_game_state)
    
    
    coordinate_old = old_game_state["self"][3]
    self.values.coordinate_history.append(coordinate_old)
    
    if e.GOT_KILLED in events:
        done = True
    else:
        done = False
    self._done = done
    action = ACTIONS.index(self_action)
    prob_a = self.prob_a
    
    max_wait = settings.EXPLOSION_TIMER
    if e.WAITED in events:
        self.values.waited_for += 1
    else:
        self.values.waited_for = 0
    if self.values.waited_for > max_wait:
        events.append(WAITED_TOO_LONG)
    
    
    if e.INVALID_ACTION in events:
        self.values.add_invalid_action()

    
    
    check_custom_events(self,events,action, feature_state_old, feature_state_new)
    
    self.values.add_event(events)
    reward = reward_from_events(self,events)
        
    
    
    # flatten feature state again
    feature_state_old = feature_state_old.flatten()
    feature_state_new = feature_state_new.flatten()
    
    

    self.values.push_data((feature_state_old,action,reward/100.0,feature_state_new,prob_a,done))

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
    if self.RANDOM_ACTION:
        events.append("RANDOM_ACTION")
        self.values.random_actions += 1
    calculate_events_and_reward(self, old_game_state, self_action, new_game_state, events)
    if not self._done:
        if TRAIN_EVERY_N_STEPS > 0:
            if self.values.global_step % TRAIN_EVERY_N_STEPS== 0:
                
                if train_in_round:
                    train_net(self)
                    self.values.log_wandb_end()
                    
            

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
    
    calculate_events_and_reward(self, last_game_state, last_action, last_game_state, events)
    
    train_steps = TRAIN_EVERY_N_STEPS
    if TRAIN_EVERY_END_OF_ROUND:
        train_steps = 1
    if self.values.global_step % train_steps == 0:

        train_net(self)
        
        # Log and reset values
        self.values.log_wandb_end()
        
        # Reset values defined in setup
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        agent_score = last_game_state["self"][1]
        if agent_score > self.best_score:
            self.best_score = agent_score
            self.logger.info(f'best score {agent_score}')
        torch.save(self.model.state_dict(), HYPER.MODEL_NAME)
            
   



def check_custom_events(self, events: List[str],action, features_old, features_new):
    # Check if agent is closer to coin
    self.values.frames_after_bomb += 1
    action = ACTIONS[action]
    if action == "BOMB":
        self.values.frames_after_bomb = 0
    
    '''
    nearest_coin_distance_old,coin_angle_old, direction_coin_old_y, direction_coin_old_x, bomb_threat_old, time_to_explode_old,\
        can_drop_bomb_old, is_next_to_opponent_old, is_on_bomb_old, should_drob_bomb_old,\
             is_in_loop_old, nearest_bomb_distance_old,\
                    blast_count_up_old, blast_count_down_old, blast_count_left_old,blast_count_right_old, \
                        danger_level_up_old, danger_level_down_old, danger_level_left_old, danger_level_right_old,*local_view_old = features_old
    
    nearest_coin_distance_new, coin_angle_new, direction_coin_new_y, direction_coin_new_x, bomb_threat_new, time_to_explode_new,\
        can_drop_bomb_new, is_next_to_opponent_new, is_on_bomb_new, should_drob_bomb_new,\
             is_in_loop_new, nearest_bomb_distance_new,\
                    blast_count_up_new, blast_count_down_new, blast_count_left_new,blast_count_right_new, \
                        danger_level_up_new, danger_level_down_new, danger_level_left_new, danger_level_right_new ,*local_view_new= features_new
    '''
    valid_direction_up_old, valid_direction_down_old,valid_direction_left_old,valid_direction_right_old, nearest_coin_distance_old, coin_angle_old, direction_coin_old_y, direction_coin_old_x, bomb_threat_old, time_to_explode_old,\
    can_drop_bomb_old, is_next_to_opponent_old, is_on_bomb_old, is_in_loop_old,\
    nearest_bomb_distance_old, blast_count_up_old, blast_count_down_old, blast_count_left_old, blast_count_right_old, \
    danger_level_up_old, danger_level_down_old, danger_level_left_old, danger_level_right_old, *local_view_old = features_old

    valid_direction_up_new, valid_direction_down_new,valid_direction_left_new,valid_direction_right_new,nearest_coin_distance_new, coin_angle_new, direction_coin_new_y, direction_coin_new_x, bomb_threat_new, time_to_explode_new,\
        can_drop_bomb_new, is_next_to_opponent_new, is_on_bomb_new, is_in_loop_new,\
        nearest_bomb_distance_new, blast_count_up_new, blast_count_down_new, blast_count_left_new, blast_count_right_new, \
        danger_level_up_new, danger_level_down_new, danger_level_left_new, danger_level_right_new, *local_view_new = features_new
            
    if "BOMB_EXPLODED" in events:
        # Check if Bomb did something usefull
        if not "CRATE_DESTROYED" in events or not "COIN_FOUND" in events or not "KILLED_OPPONENT" in events:
            events.append("BOMB_WAS_USELESS")
    #Check if agent chose valid direction
    if action == "UP" and not valid_direction_up_old:
        events.append("AGENT_CHOSE_INVALID_DIRECTION")
    if action == "DOWN" and not valid_direction_down_old:
        events.append("AGENT_CHOSE_INVALID_DIRECTION")
    if action == "LEFT" and not valid_direction_left_old:
        events.append("AGENT_CHOSE_INVALID_DIRECTION")
    if action == "RIGHT" and not valid_direction_right_old:
        events.append("AGENT_CHOSE_INVALID_DIRECTION")
    
    
    if nearest_coin_distance_new < nearest_coin_distance_old:
        events.append(CLOSER_TO_COIN)
    else: 
        events.append(FURTHER_FROM_COIN)
        
    # Check if bomb is escapable
    if is_on_bomb_new and not is_on_bomb_old:
        events.append(ESCAPABLE_BOMB)
    
    # Check if Agent registered bomb threat and escaped
    if bomb_threat_old and not bomb_threat_new:
        events.append(ESCAPED_BOMB)
    # Check if Agent registered bomb threat and went towards it
    if nearest_bomb_distance_old > nearest_bomb_distance_new:
        events.append(GOING_TOWARDS_BOMB)
    # Check if Agent registered bomb threat and went away from it
    if nearest_bomb_distance_old < nearest_bomb_distance_new:
        events.append(GOING_AWAY_FROM_BOMB)
    
    coordinate_old = self.values.coordinate_history[-1]
    if coordinate_old in self.values.coordinate_history:
        events.append(IS_IN_LOOP)
    
    # Check if agent reduced blast count in certain direction
    if blast_count_up_new < blast_count_up_old:
        # Check if agent took action to reduce blast count
        if action == "DOWN":
            events.append(BLAST_COUNT_UP_DECREASED)
    if blast_count_down_new < blast_count_down_old:
        # Check if agent took action to reduce blast count
        if action == "UP":
            events.append(BLAST_COUNT_DOWN_DECREASED)
            
    if blast_count_left_new < blast_count_left_old:
        # Check if agent took action to reduce blast count
        if action == "RIGHT":
            events.append(BLAST_COUNT_LEFT_DECREASED)
    
    if blast_count_right_new < blast_count_right_old:
        # Check if agent took action to reduce blast count
        if action == "LEFT":
            events.append(BLAST_COUNT_RIGHT_DECREASED)
            
    # Check if Agent collected coin while bomb was placed
    if self.values.frames_after_bomb < 5 and e.COIN_COLLECTED in events:
        events.append(DROPPED_BOMB_AND_COLLECTED_COIN_MEANWHILE)
    # Check if agent went into bomb radius and died
    if e.GOT_KILLED:
        if danger_level_up_old == 1 and action == "UP":
            events.append(WENT_INTO_BOMB_RADIUS_AND_DIED)
        if danger_level_down_old == 1 and action == "DOWN":
            events.append(WENT_INTO_BOMB_RADIUS_AND_DIED)
        if danger_level_left_old == 1 and action == "LEFT":
            events.append(WENT_INTO_BOMB_RADIUS_AND_DIED)
        if danger_level_right_old == 1 and action == "RIGHT":
            events.append(WENT_INTO_BOMB_RADIUS_AND_DIED)
            
    if DEBUG_EVENTS:
        self.logger.info(f'Events: {events}')
            
    
        
    

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    # Count number of invalid actions
    if e.INVALID_ACTION in events:
        self.invalid_actions += 1
    
    coin_rewards_loot_crate = {
        e.COIN_COLLECTED: 80,
        CLOSER_TO_COIN: 5,
        FURTHER_FROM_COIN:-10,
        e.INVALID_ACTION:-20,
        e.CRATE_DESTROYED: 20,
        e.COIN_FOUND: 15,
        WAITED_TOO_LONG:-20,
        e.KILLED_SELF:-150,
        ESCAPED_BOMB: 28,
        GOING_TOWARDS_BOMB:-25,
        GOING_AWAY_FROM_BOMB: 25,
        IS_IN_LOOP: -20,
        GOT_OUT_OF_LOOP: 10,#not sure if this is working
        IN_BLAST_RADIUS:-50,
        BLAST_COUNT_UP_DECREASED: 5,
        BLAST_COUNT_DOWN_DECREASED: 5,
        BLAST_COUNT_LEFT_DECREASED: 5,
        BLAST_COUNT_RIGHT_DECREASED: 5,
        WENT_INTO_BOMB_RADIUS_AND_DIED: -25,
        AGENT_CHOSE_INVALID_DIRECTION: -50, 
        BOMB_WAS_USELESS:-70,
    }
    coin_rewards = coin_rewards_loot_crate
    reward_sum = 0
    for event in events:
        if event in coin_rewards:
            reward_sum += coin_rewards[event]
    
    return reward_sum

