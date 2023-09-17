from collections import namedtuple, deque, Counter
import numpy as np
import pickle
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

import events as e
from .callbacks import state_to_features, ACTIONS, LOG_WANDB, DEBUG_EVENTS, LOG_TO_FILE

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

WANDB_NAME = "COIN_COLLECTOR_WITHOUT_DIRECTION"
WANDB_FLAG = LOG_WANDB


DEBUG_EVENTS =  DEBUG_EVENTS

import settings

log_to_file = LOG_TO_FILE

train_in_round = False 
TRAIN_EVERY_N_STEPS = 30
TRAIN_EVERY_END_OF_ROUND = True
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
        #wandb.save("logs/PPOLD.log") if WANDB_FLAG else None
    
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
        wandb.watch(self.model)
    #self.global_step = 0

    #self.loss_history = []
    #self.reward_history = []
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=HYPER.learning_rate)
    self.invalid_actions = 0
    
    self.values = Values(   logger=self.logger)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.logger.info(f'Using device: {self.device}')
    self.model = self.model.to(self.device)
    self.best_score = 0
   
          
def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.values.make_batch()
        #Put data on device
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
            
            #self.loss_history.append(loss.mean().item())
            self.optimizer.step()  
            self.values.add_loss(loss.mean().item())
        

#----------TODO change this to custom one
#from .reward_shaping import custom_rewards, reward_coin_distance, check_placed_bomb, check_blast_radius, CLOSER_TO_COIN, FURTHER_FROM_COIN, ESCAPABLE_BOMB, BOMB_DESTROYS_CRATE, WAITED_TOO_LONG, IN_BLAST_RADIUS
WAITED_TOO_LONG = "WAITED_TOO_LONG"

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
    
    
    
    
    """
    
    
    
    # Check if agent is closer to closest coin
    # reshape feature state to square grid 
    feature_state_old = feature_state_old.reshape(17,17)
    feature_state_new = feature_state_new.reshape(17,17)
    
    # Get coordinates of agent
    agent_pos_old = np.where(feature_state_old == 5)
    agent_pos_new = np.where(feature_state_new == 5)
    
    
    
    # Get coordinates of coins
    coin_positions_old = np.where(feature_state_old == 1.5)
    coin_positions_new = np.where(feature_state_new == 1.5)

    # get closest coin
    distances_to_coin = [np.abs(x - agent_pos_old[0]) + np.abs(y - agent_pos_old[1]) for x,y in zip(coin_positions_old[0],coin_positions_old[1])]
    closest_coin_old = np.argmin(distances_to_coin)
    closest_coin_x_y = (coin_positions_old[0][closest_coin_old],coin_positions_old[1][closest_coin_old])

    # Get distance to closest coin
    distance_to_closest_coin_old = np.abs(closest_coin_x_y[0] - agent_pos_old[0]) + np.abs(closest_coin_x_y[1] - agent_pos_old[1])
    distance_to_closest_coin_new = np.abs(closest_coin_x_y[0] - agent_pos_new[0]) + np.abs(closest_coin_x_y[1] - agent_pos_new[1])
    
    # Check if agent is closer to coin
    if distance_to_closest_coin_new < distance_to_closest_coin_old:
        events.append(CLOSER_TO_COIN)
    else:
        events.append(FURTHER_FROM_COIN)
    
    
    
    """
    
    
    
    
    
    
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
   
   
    
    
    if not self._done:
        if TRAIN_EVERY_N_STEPS > 0:
            if self.values.global_step % TRAIN_EVERY_N_STEPS== 0:
                
                if train_in_round:
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
    
    calculate_events_and_reward(self, last_game_state, last_action, last_game_state, events)
    
    # 
    train_steps = TRAIN_EVERY_N_STEPS
    #train_steps = 1 # train after each round
    if TRAIN_EVERY_END_OF_ROUND:
        train_steps = 1
    if self.values.global_step % train_steps == 0:

        train_net(self)
        
        # Log and reset values
        self.values.log_wandb_end()
        
        # Reset values defined in setup
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        # Store the model if it is the best one
        agent_score = last_game_state["self"][1]
        if agent_score > self.best_score:
            self.best_score = agent_score
            self.logger.info(f'best score {agent_score}')
        with open(HYPER.MODEL_NAME, "wb") as file:
            pickle.dump(self.model, file)
            
    #clear log file
    with open("logs/PPOLD.log", "w") as file:
        file.write("")



REPEATING_ACTIONS = "REPEATING_ACTIONS"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"
ESCAPABLE_BOMB = "ESCAPABLE_BOMB"
BOMB_DESTROYS_CRATE = "BOMB_DESTROYS_CRATE"

TOOK_DIRECTION_TOWARDS_TARGET = "TOOK_DIRECTION_TOWARDS_TARGET"
TOOK_DIRECTION_AWAY_FROM_TARGET = "TOOK_DIRECTION_AWAY_FROM_TARGET"

IS_IN_LOOP = "IS_IN_LOOP"
GOT_OUT_OF_LOOP = "GOT_OUT_OF_LOOP"

# Wheter dropped bomb when he should have
DROPPED_BOMB_WHEN_SHOULDNT = "DROPPED_BOMB_WHEN_SHOULDNT"
# Wheter did not drop bomb when he should have
DIDNT_DROP_BOMB_WHEN_SHOULD = "DIDNT_DROP_BOMB_WHEN_SHOULD"

# Wheter agent successfully placed bomb
DROPPED_BOMB_WHEN_SHOULD_BUT_STAYED = "DROPPED_BOMB_WHEN_SHOULD_BUT_STAYED"
DROPPED_BOMB_WHEN_SHOULD_AND_MOVED = "DROPPED_BOMB_WHEN_SHOULD_AND_MOVED"
# Wheter agent is in blast radius
IN_BLAST_RADIUS = "IN_BLAST_RADIUS"

ESCAPED_BOMB = "ESCAPED_BOMB"
GOING_TOWARDS_BOMB = "GOING_TOWARDS_BOMB"
GOING_AWAY_FROM_BOMB = "GOING_AWAY_FROM_BOMB"


#Check if agent redcues blast count in certain direction
BLAST_COUNT_UP_DECREASED = "BLAST_COUNT_UP_DECREASED"
BLAST_COUNT_DOWN_DECREASED = "BLAST_COUNT_DOWN_DECREASED"
BLAST_COUNT_LEFT_DECREASED = "BLAST_COUNT_LEFT_DECREASED"
BLAST_COUNT_RIGHT_DECREASED = "BLAST_COUNT_RIGHT_DECREASED"

WENT_INTO_BOMB_RADIUS_AND_DIED = "WENT_INTO_BOMB_RADIUS_AND_DIED"
DROPPED_BOMB_AND_COLLECTED_COIN_MEANWHILE = "DROPPED_BOMB_AND_COLLECTED_COIN_MEANWHILE"



def check_custom_events(self, events: List[str],action, features_old, features_new):
    # Check if agent is closer to coin
    self.values.frames_after_bomb += 1
    action = ACTIONS[action]
    if action == "BOMB":
        self.values.frames_after_bomb = 0
    
    nearest_coin_distance_old,coin_angle_old, direction_coin_old, bomb_threat_old, time_to_explode_old,\
        can_drop_bomb_old, is_next_to_opponent_old, is_on_bomb_old, should_drob_bomb_old,\
             is_in_loop_old, nearest_bomb_distance_old,\
                    blast_count_up_old, blast_count_down_old, blast_count_left_old,blast_count_right_old, \
                        danger_level_up_old, danger_level_down_old, danger_level_left_old, danger_level_right_old,*local_view_old = features_old
    
    nearest_coin_distance_new, coin_angle_new, direction_coin_new, bomb_threat_new, time_to_explode_new,\
        can_drop_bomb_new, is_next_to_opponent_new, is_on_bomb_new, should_drob_bomb_new,\
             is_in_loop_new, nearest_bomb_distance_new,\
                    blast_count_up_new, blast_count_down_new, blast_count_left_new,blast_count_right_new, \
                        danger_level_up_new, danger_level_down_new, danger_level_left_new, danger_level_right_new ,*local_view_new= features_new
                
                
    
    
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

    '''
    # Check if Agent took direction towards target: direction_to_target = [UP, DOWN, LEFT, RIGHT]
    if (action == "UP" and direction_to_target_old_UP == 1) or (action == "DOWN" and direction_to_target_old_DOWN == 1)\
        or (action == "LEFT" and direction_to_target_old_LEFT == 1) or (action == "RIGHT" and direction_to_target_old_RIGHT == 1):
        TARGET_FLAG = True
        events.append(TOOK_DIRECTION_TOWARDS_TARGET)
    else:
        TARGET_FLAG = False
        # Check if direction to target is available
        _sum = direction_to_target_old_UP + direction_to_target_old_DOWN + direction_to_target_old_LEFT + direction_to_target_old_RIGHT
        if _sum != 0:
            events.append(TOOK_DIRECTION_AWAY_FROM_TARGET) # TODO: check if this is correct
    '''
    
    # Check if Agent registered bomb threat and escaped
    if bomb_threat_old and not bomb_threat_new:
        events.append(ESCAPED_BOMB)
    # Check if Agent registered bomb threat and went towards it
    if nearest_bomb_distance_old > nearest_bomb_distance_new:
        events.append(GOING_TOWARDS_BOMB)
    # Check if Agent registered bomb threat and went away from it
    if nearest_bomb_distance_old < nearest_bomb_distance_new:
        events.append(GOING_AWAY_FROM_BOMB)
    
    # TODO: check if this works,
    
    coordinate_old = self.values.coordinate_history[-1]
    if coordinate_old in self.values.coordinate_history:
        events.append(IS_IN_LOOP)

     
     #TODO CHECK THIS
        
    # Check if agent dropped bomb when he should have
    if not should_drob_bomb_old and action=="BOMB":
        events.append(DROPPED_BOMB_WHEN_SHOULDNT)
        
    # Check if agent did not drop bomb when he should have
    if should_drob_bomb_old and action !="BOMB":
        events.append(DIDNT_DROP_BOMB_WHEN_SHOULD)
        
    # Check if agent successfully placed bomb
    if should_drob_bomb_old and action=="BOMB":
        if nearest_bomb_distance_new == 0:
            events.append(DROPPED_BOMB_WHEN_SHOULD_BUT_STAYED)
        if nearest_bomb_distance_new > 0:
            events.append(DROPPED_BOMB_WHEN_SHOULD_AND_MOVED)
        
    blast_count_old = [blast_count_up_old, blast_count_down_old, blast_count_left_old, blast_count_right_old]
    blast_count_new = [blast_count_up_new, blast_count_down_new, blast_count_left_new, blast_count_right_new]
    
    danger_level_old = [danger_level_up_old, danger_level_down_old, danger_level_left_old, danger_level_right_old]
    danger_level_new = [danger_level_up_new, danger_level_down_new, danger_level_left_new, danger_level_right_new]
    
    
    # TODO hier punishen wenn agent bombe placed aber sich nicht rechtzeitig bewegt, oben nochmal abchecekn
    #---------------------------------------_!
    
        
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
        # Check if agent got killed by walking into bomb radius,
        # i.e check danger level and if agent took action towards it
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
    coin_rewards_loot_crate = {
        e.COIN_COLLECTED: 20,
        #FURTHER_FROM_COIN:-10,
        CLOSER_TO_COIN: 19,
        #FURTHER_FROM_COIN:-10,
        e.INVALID_ACTION:-10,
        e.CRATE_DESTROYED: 20,
        e.COIN_FOUND: 15,
        WAITED_TOO_LONG:-20,
        DROPPED_BOMB_WHEN_SHOULDNT:-150,
        DIDNT_DROP_BOMB_WHEN_SHOULD:-25,
        DROPPED_BOMB_WHEN_SHOULD_AND_MOVED: 20,
        DROPPED_BOMB_WHEN_SHOULD_BUT_STAYED: -5,
        e.KILLED_SELF:-25,
        ESCAPED_BOMB: 28,
        GOING_TOWARDS_BOMB:-5,
        GOING_AWAY_FROM_BOMB: 10,
        #TOOK_DIRECTION_TOWARDS_TARGET: 20,
        #TOOK_DIRECTION_AWAY_FROM_TARGET: -25,
        IS_IN_LOOP: -10,
        #GOT_OUT_OF_LOOP: 10,#not sure if this is working
        IN_BLAST_RADIUS:-50,
        BLAST_COUNT_UP_DECREASED: 25,
        BLAST_COUNT_DOWN_DECREASED: 25,
        BLAST_COUNT_LEFT_DECREASED: 25,
        BLAST_COUNT_RIGHT_DECREASED: 25,
        WENT_INTO_BOMB_RADIUS_AND_DIED: -25,
        DROPPED_BOMB_AND_COLLECTED_COIN_MEANWHILE: 25,
        e.SURVIVED_ROUND:150,
    }
    
    # make this a global dict
    

    
    coin_rewards = coin_rewards_loot_crate
    reward_sum = -5
    reward_sum = -self.values.global_step/100
    for event in events:
        if event in coin_rewards:
            reward_sum += coin_rewards[event]
        
    #self.reward_history.append(reward_sum)
    return reward_sum
    #reward_sum =  custom_rewards(self,events)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
