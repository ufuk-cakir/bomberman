from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, LOG_WANDB, CONTINUE_TRAINING

from .Qmodel import QNet, Memory, Transition, HYPER, ACTIONS
import torch
import torch.optim as optim
import torch.nn.functional as F



#----------
import wandb

#set scenario for rewards
scenario = "loot-crate"

#count steps in current batch
#batch_count = 0

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"




def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Setup optimizer
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=HYPER.LR, amsgrad=True)
    self.memory = Memory(10000)
    self.score = 0
    self.reward_history = []
    self.loss_history = []
    self.events_history = []
    self.avtion_history = []
    self.frames_after_bomb = 0
    #logging variables
    self.log_crates = 0
    self.log_coins = 0

    




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
    self.events_history.append(events)
    self.avtion_history.append(self_action)
    
    feature_state_old = state_to_features(self,old_game_state)
    feature_state_new = state_to_features(self,new_game_state)
    
    
    if e.GOT_KILLED in events:
        done = True
    else:
        done = False
    action = ACTIONS.index(self_action)

    check_custom_events(self, events, action, feature_state_old, feature_state_new)

    reward = reward_from_events(self,events)

    if done:
        feature_state_new = None
    else:
        feature_state_new = torch.tensor(feature_state_new, device=self.device, dtype=torch.float).unsqueeze(0)
    
    feature_state_old = torch.tensor(feature_state_old, device=self.device, dtype=torch.float).unsqueeze(0)
    action = torch.tensor([[action]], device=self.device, dtype=torch.long)    
    reward = torch.tensor([reward], device=self.device)
    
    # Store transition in memory
    self.memory.push(feature_state_old, action, feature_state_new, reward)
    self.score += reward
    
    # Optimize Fulley if batch size is reached, otherwise single
    #batch_count =+ 1
    if len(self.memory) % HYPER.BATCH_SIZE == 1: #batch_count == HYPER.BATCH_SIZE: #len(self.memory) % HYPER.BATCH_SIZE == 1:
        optimize_model(self)
        #batch_count = 0
        
    else:
       optimize_model_single(self, (feature_state_old, action, feature_state_new, reward))   
    
    


def optimize_model_single(self, transition):
    self.logger.info(f'Single Optimization...')
    state, action, next_state, reward = transition

    # Compute Q(s_t, a) and V(s_{t+1})
    state_value = self.policy_net(state).gather(1, action)
    next_state_value = torch.zeros(1, device=self.device)
    
    if next_state is not None:
        next_state_value = self.target_net(next_state).max(1)[0].detach()

    # Compute the expected Q value using bootstrapping
    expected_state_action_value = (next_state_value * HYPER.GAMMA) + reward

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_value, expected_state_action_value.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    # Soft update of target network
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = HYPER.TAU * policy_net_state_dict[key] + (1 - HYPER.TAU) * target_net_state_dict[key]
    self.target_net.load_state_dict(target_net_state_dict)
    
    # Watch the gradients of model parameters
    if(LOG_WANDB):
        #wandb.log({"loss": loss.item(), "cumulative_reward": self.score})
        wandb.log({"loss": loss.item()})
    #     #wandb.watch(self.policy_net)
    
    # Compute gradient statistics
        for name, param in self.policy_net.named_parameters():
            if param.grad is not None:
                wandb.log({f"gradient/{name}": param.grad.norm()})



def optimize_model(self):
    if len(self.memory) < HYPER.BATCH_SIZE:
        self.logger.info(f'Not enough samples in memory to optimize model.')
        return
    self.logger.info(f'Optimizing model Fulle...')
    #self.logger.info(f'Optimizing model...')
    transitions = self.memory.sample(HYPER.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
 
    state_batch = torch.cat(batch.state)
    state_batch = state_batch.reshape(HYPER.BATCH_SIZE, HYPER.N_FEATURES)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
   
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(HYPER.BATCH_SIZE, device=self.device)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * HYPER.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()
    # Soft update of target network
    # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = HYPER.TAU * policy_net_state_dict[key] + (1 - HYPER.TAU) * target_net_state_dict[key]
    self.target_net.load_state_dict(target_net_state_dict)
    
    # Watch the gradients of model parameters
    #if LOG_WANDB:
    #    wandb.log({"loss": loss.item(), "cumulative_reward": self.score})
        #wandb.watch(self.policy_net)
    
    # Compute gradient statistics
        # for name, param in self.policy_net.named_parameters():
        #     if param.grad is not None:
        #         wandb.log({f"gradient/{name}": param.grad.norm()})



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
    # Add final events to game events TODO: REWRITE
    state_final = state_to_features(self,last_game_state)
    state = torch.tensor(state_final, device=self.device, dtype=torch.float).unsqueeze(0)
    final_state = None
    action = ACTIONS.index(last_action)
    action = torch.tensor([[action]], device=self.device, dtype=torch.long)
    check_custom_events(self, events, action, state_final, state_final)
    reward = reward_from_events(self,events)
    reward = torch.tensor([reward], device=self.device)
    self.memory.push(state, action, final_state, reward)
    
    # do final optimization step
    optimize_model(self)
    
    with open("policy_net.pt", "wb") as file:
        pickle.dump(self.policy_net, file)
    with open("target_net.pt", "wb") as file:
        pickle.dump(self.target_net, file)
    
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.logger.info(f'End of round with cumulative reward {self.score}')
    self.logger.info("Saved model.")
    
    if LOG_WANDB:
        wandb.log({"final_cumulative_reward": self.score}) 
        wandb.log({"coins collected": self.log_coins})
        wandb.log({"crates destroyed": self.log_crates})
        wandb.log({"steps survived": len(self.memory)})
    self.score = 0
    self.memory = Memory(10000)
    self.log_crates = 0
    self.log_coins = 0
    

#Custom Rewards


REPEATING_ACTIONS = "REPEATING_ACTIONS"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"
ESCAPABLE_BOMB = "ESCAPABLE_BOMB"
CRATE_DESTROYED_AND_SURVIVED = "CRATE_DESTROYED_AND_SURVIVED"

TOOK_DIRECTION_TOWARDS_TARGET = "TOOK_DIRECTION_TOWARDS_TARGET"
TOOK_DIRECTION_AWAY_FROM_TARGET = "TOOK_DIRECTION_AWAY_FROM_TARGET"
NO_TARGET_DIRECTION = "NO_TARGET_DIRECTION"

IS_IN_LOOP = "IS_IN_LOOP"
GOT_OUT_OF_LOOP = "GOT_OUT_OF_LOOP"

# Wheter dropped bomb when he should have
DROPPED_BOMB_WHEN_SHOULDNT = "DROPPED_BOMB_WHEN_SHOULDNT"
# Wheter did not drop bomb when he should have
DIDNT_DROP_BOMB_WHEN_SHOULD = "DIDNT_DROP_BOMB_WHEN_SHOULD"

# Wheter agent successfully placed bomb
DROPPED_BOMB_WHEN_SHOULD = "DROPPED_BOMB_WHEN_SHOULD"
# Wheter agent is in blast radius
IN_BLAST_RADIUS = "IN_BLAST_RADIUS"
IN_BLAST_RADIUS_AND_NOT_TRYING_TO_ESCAPE = "IN_BLAST_RADIUS_AND_NOT_TRYING_TO_ESCAPE"

ESCAPED_BOMB = "ESCAPED_BOMB"
SURVIVED_BOMB = "SURVIVED_BOMB"
GOING_TOWARDS_BOMB = "GOING_TOWARDS_BOMB"
GOING_AWAY_FROM_BOMB = "GOING_AWAY_FROM_BOMB"


#Check if agent redcues blast count in certain direction
BLAST_COUNT_UP_DECREASED = "BLAST_COUNT_UP_DECREASED"
BLAST_COUNT_DOWN_DECREASED = "BLAST_COUNT_DOWN_DECREASED"
BLAST_COUNT_LEFT_DECREASED = "BLAST_COUNT_LEFT_DECREASED"
BLAST_COUNT_RIGHT_DECREASED = "BLAST_COUNT_RIGHT_DECREASED"

#check if agent waited in blast radius
WAITED_IN_BLAST_RADIUS = "WAITED_IN_BLAST_RADIUS"

WENT_INTO_BOMB_RADIUS_AND_DIED = "WENT_INTO_BOMB_RADIUS_AND_DIED"
DROPPED_BOMB_AND_COLLECTED_COIN_MEANWHILE = "DROPPED_BOMB_AND_COLLECTED_COIN_MEANWHILE"

def check_custom_events(self, events: List[str],action, features_old, features_new):
    # Check if agent is closer to coin
    self.frames_after_bomb += 1
    action = ACTIONS[action]
    if action == "BOMB":
        self.frames_after_bomb = 0

    nearest_coin_distance_old, is_dead_end_old, bomb_threat_old, time_to_explode_old,\
        can_drop_bomb_old, is_next_to_opponent_old, is_on_bomb_old, should_drob_bomb_old,\
            escape_route_available_old, direction_to_target_old_UP, direction_to_target_old_DOWN,\
                direction_to_target_old_LEFT, direction_to_target_old_RIGHT, is_in_loop_old, nearest_bomb_distance_old,\
                    blast_count_up_old, blast_count_down_old, blast_count_left_old,blast_count_right_old, \
                        danger_level_up_old, danger_level_down_old, danger_level_left_old, danger_level_right_old, *local_map_old = features_old
    
    nearest_coin_distance_new, is_dead_end_new, bomb_threat_new, time_to_explode_new,\
        can_drop_bomb_new, is_next_to_opponent_new, is_on_bomb_new, should_drob_bomb_new,\
            escape_route_available_new, direction_to_target_new_UP, direction_to_target_new_DOWN,\
                direction_to_target_new_LEFT, direction_to_target_new_RIGHT, is_in_loop_new, nearest_bomb_distance_new,\
                    blast_count_up_new, blast_count_down_new, blast_count_left_new,blast_count_right_new, \
                        danger_level_up_new, danger_level_down_new, danger_level_left_new, danger_level_right_new,*local_map_new = features_new
                
    # nearest_coin_distance_old, is_dead_end_old, bomb_threat_old, time_to_explode_old,\
    #     can_drop_bomb_old, is_next_to_opponent_old, is_on_bomb_old, should_drob_bomb_old,\
    #         escape_route_available_old, is_in_loop_old, nearest_bomb_distance_old,\
    #                 blast_count_up_old, blast_count_down_old, blast_count_left_old,blast_count_right_old, \
    #                     danger_level_up_old, danger_level_down_old, danger_level_left_old, danger_level_right_old, *local_map_old = features_old
    
    # nearest_coin_distance_new, is_dead_end_new, bomb_threat_new, time_to_explode_new,\
    #     can_drop_bomb_new, is_next_to_opponent_new, is_on_bomb_new, should_drob_bomb_new,\
    #         escape_route_available_new, is_in_loop_new, nearest_bomb_distance_new,\
    #                 blast_count_up_new, blast_count_down_new, blast_count_left_new,blast_count_right_new, \
    #                     danger_level_up_new, danger_level_down_new, danger_level_left_new, danger_level_right_new,*local_map_new = features_new           
    
    
    # Feature list: [nearest_coin_distance, is_dead_end, bomb_threat, time_to_explode, can_drop_bomb, 
    # is_next_to_opponent, is_on_bomb, crates_nearby, escape_route_available, direction_to_target, is_in_loop, ignore_others_timer_normalized]
    if nearest_coin_distance_new < nearest_coin_distance_old:
        events.append(CLOSER_TO_COIN)
    else: 
        events.append(FURTHER_FROM_COIN)
        
    # Check if bomb is escapable
    if is_on_bomb_new and not is_on_bomb_old:
        events.append(ESCAPABLE_BOMB)
    #for logging count crates destroyed and if coin collected
    crates = events.count(e.CRATE_DESTROYED)
    self.log_crates += crates
    if e.COIN_COLLECTED in events:
        self.log_coins += 1
    # Check if crate destroyed and survived
    if not e.KILLED_SELF in events:
        for i in range(crates):
            events.append(CRATE_DESTROYED_AND_SURVIVED)
        

    # # Check if Agent took direction towards target: direction_to_target = [UP, DOWN, LEFT, RIGHT]
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
        else:
            events.append(NO_TARGET_DIRECTION)
    
    # Check if Agent registered bomb threat and escaped
    if bomb_threat_old and not bomb_threat_new:
        events.append(ESCAPED_BOMB)
    # Check if Agent registered bomb threat and went towards it
    if nearest_bomb_distance_old > nearest_bomb_distance_new:
        events.append(GOING_TOWARDS_BOMB)
    # Check if Agent registered bomb threat and went away from it
    if nearest_bomb_distance_old < nearest_bomb_distance_new:
        events.append(GOING_AWAY_FROM_BOMB)
    #Check if Agent stayed in blast radius without increasing the distance to the bomb
    if nearest_bomb_distance_new <= nearest_bomb_distance_old and bomb_threat_old and bomb_threat_new:
        events.append(IN_BLAST_RADIUS_AND_NOT_TRYING_TO_ESCAPE)
    #Check if Agent went into Blast Radius
    if (bomb_threat_old and action == "WAIT") or (bomb_threat_old and e.INVALID_ACTION in events):
        events.append(WAITED_IN_BLAST_RADIUS)


    
    # # TODO: check if this works,
    # if is_in_loop_new:
    #     events.append(IS_IN_LOOP)
    # if not is_in_loop_new and is_in_loop_old:
    #     events.append(GOT_OUT_OF_LOOP)
     
     #TODO CHECK THIS
        
    # Check if agent dropped bomb when he should have
    if not should_drob_bomb_old and action=="BOMB":
        events.append(DROPPED_BOMB_WHEN_SHOULDNT)
        
    # Check if agent did not drop bomb when he should have
    if should_drob_bomb_old and action !="BOMB":
        events.append(DIDNT_DROP_BOMB_WHEN_SHOULD)
        
    # Check if agent successfully placed bomb
    if should_drob_bomb_old and action=="BOMB":
        events.append(DROPPED_BOMB_WHEN_SHOULD)
        
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
    if self.frames_after_bomb < 5 and e.COIN_COLLECTED in events:
        events.append(DROPPED_BOMB_AND_COLLECTED_COIN_MEANWHILE)
    
    # Check if agent went into bomb radius and died
    if e.GOT_KILLED in events:
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


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    game_rewards = {

    'empty': {
    e.COIN_COLLECTED: 0.0,
    e.KILLED_OPPONENT: 0.0,
    e.KILLED_SELF: 0.0,
    e.INVALID_ACTION: 0.0,
    e.WAITED: 0.0,
    e.MOVED_LEFT: 0.0,
    e.MOVED_RIGHT: 0.0,
    e.MOVED_UP: 0.0,
    e.MOVED_DOWN: 0.0,
    e.BOMB_DROPPED: 0.0,
    e.BOMB_EXPLODED: 0.0,
    e.CRATE_DESTROYED: 0.0,
    e.COIN_FOUND: 0.0,
    e.SURVIVED_ROUND: 0.0,
    e.GOT_KILLED: 0.0},

    'coin-heaven': {
    e.COIN_COLLECTED: 15.0,
    e.KILLED_OPPONENT: 0.0,
    e.KILLED_SELF: -10.0,
    e.INVALID_ACTION: -1.0,
    e.WAITED: -0.5,
    e.MOVED_LEFT: -0.1,
    e.MOVED_RIGHT: -0.1,
    e.MOVED_UP: -0.1,
    e.MOVED_DOWN: -0.1,
    e.BOMB_DROPPED: -7.0,
    e.BOMB_EXPLODED: 0.0,
    e.CRATE_DESTROYED: 0.0,
    e.COIN_FOUND: 0.0,
    e.SURVIVED_ROUND: 1.0,
    e.GOT_KILLED: 0.0},
    
    #performs way worse than before
    'loot-crate': {
    e.COIN_COLLECTED: 15.0,
    e.KILLED_SELF: -20.0,
    e.INVALID_ACTION: -0.5,
    e.WAITED: -0.1,
    e.MOVED_LEFT: -0.1,
    e.MOVED_RIGHT: -0.1,
    e.MOVED_UP: -0.1,
    e.MOVED_DOWN: -0.1,
    e.BOMB_DROPPED: 0.0,
    e.BOMB_EXPLODED: 0.0,
    e.CRATE_DESTROYED: 0.0,
    CRATE_DESTROYED_AND_SURVIVED: 4.0,
    e.COIN_FOUND: 0.0,
    e.SURVIVED_ROUND: 30.0,
    DROPPED_BOMB_WHEN_SHOULD: 0.5,
    DROPPED_BOMB_WHEN_SHOULDNT:-1.0,
    DIDNT_DROP_BOMB_WHEN_SHOULD:-0.5,
    ESCAPED_BOMB: 0.5,
    GOING_TOWARDS_BOMB:-0.2,
    GOING_AWAY_FROM_BOMB: 0.2,
    TOOK_DIRECTION_TOWARDS_TARGET: 1.0,
    TOOK_DIRECTION_AWAY_FROM_TARGET: -1.5,
    # IS_IN_LOOP: -0.2,
    # GOT_OUT_OF_LOOP: 0.2,#not sure if this is working
    IN_BLAST_RADIUS:-0.5,
    BLAST_COUNT_UP_DECREASED: 0.3,
    BLAST_COUNT_DOWN_DECREASED: 0.3,
    BLAST_COUNT_LEFT_DECREASED: 0.3,
    BLAST_COUNT_RIGHT_DECREASED: 0.3,
    WENT_INTO_BOMB_RADIUS_AND_DIED: -5.0,
    #DROPPED_BOMB_AND_COLLECTED_COIN_MEANWHILE: 2.5,
    IN_BLAST_RADIUS_AND_NOT_TRYING_TO_ESCAPE: -2.0,
    #WAITED_IN_BLAST_RADIUS: -1.0,
    NO_TARGET_DIRECTION: 0.0
    },

    'classic': {
    e.COIN_COLLECTED: 10.0,
    e.KILLED_SELF: -10.0,
    e.INVALID_ACTION: -0.5,
    e.WAITED: 0.1,
    e.MOVED_LEFT: 0.1,
    e.MOVED_RIGHT: 0.1,
    e.MOVED_UP: 0.1,
    e.MOVED_DOWN: 0.1,
    e.BOMB_DROPPED: 0.5,
    e.BOMB_EXPLODED: 0.5,
    e.CRATE_DESTROYED: 0.0,
    CRATE_DESTROYED_AND_SURVIVED: 4.0,
    e.COIN_FOUND: 2.0,
    e.SURVIVED_ROUND: 30.0,
    DROPPED_BOMB_WHEN_SHOULDNT:-1.0,
    DIDNT_DROP_BOMB_WHEN_SHOULD:-1.0,
    ESCAPED_BOMB: 1.0,
    GOING_TOWARDS_BOMB:-1.0,
    GOING_AWAY_FROM_BOMB: 1.0,
    TOOK_DIRECTION_TOWARDS_TARGET: 2.0,
    TOOK_DIRECTION_AWAY_FROM_TARGET: -2.0,
    # IS_IN_LOOP: -0.2,
    # GOT_OUT_OF_LOOP: 0.2,#not sure if this is working
    IN_BLAST_RADIUS:-1.0,
    BLAST_COUNT_UP_DECREASED: 1.0,
    BLAST_COUNT_DOWN_DECREASED: 1.0,
    BLAST_COUNT_LEFT_DECREASED: 1.0,
    BLAST_COUNT_RIGHT_DECREASED: 1.0,
    WENT_INTO_BOMB_RADIUS_AND_DIED: -5.0,
    #DROPPED_BOMB_AND_COLLECTED_COIN_MEANWHILE: 2.5,
    IN_BLAST_RADIUS_AND_NOT_TRYING_TO_ESCAPE: -3.0,
    WAITED_IN_BLAST_RADIUS: -2.0,
    },

}
    reward_sum = 0 # default reward for surviving
    for event in events:
        if event in game_rewards[scenario]:
            reward_sum += game_rewards[scenario][event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    # if LOG_WANDB:
    #     wandb.log({"reward": reward_sum})
    return reward_sum

