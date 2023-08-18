from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features 

from .Qmodel import QNet, Memory, Transition,HYPER, ACTIONS
import torch
import torch.optim as optim
import torch.nn.functional as F




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
    feature_state_old = state_to_features(self,old_game_state)
    feature_state_new = state_to_features(self,new_game_state)
    
    
    if e.GOT_KILLED in events:
        done = True
    else:
        done = False
    action = ACTIONS.index(self_action)
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
    
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # Idea: Add your own events to hand out rewards
    #if ...:
     #   events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    self.logger.info(f'Starting to train...')
    optimize_model(self)
    
    # Soft update of target network
    # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = HYPER.TAU * policy_net_state_dict[key] + (1 - HYPER.TAU) * target_net_state_dict[key]
    self.target_net.load_state_dict(target_net_state_dict)
    





def optimize_model(self):
    self.logger.info(f'Optimizing model...')
    if len(self.memory) < HYPER.BATCH_SIZE:
        return
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

    # Store the model
    
    with open("policy_net.pt", "wb") as file:
        pickle.dump(self.policy_net, file)
    with open("target_net.pt", "wb") as file:
        pickle.dump(self.target_net, file)
    self.logger.info(f'End of round with cumulative reward {self.score}')
    self.logger.info("Saved model.")

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -.6,  # idea: the custom event is bad
        e.INVALID_ACTION: -10,
        e.WAITED: -1,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.BOMB_DROPPED: 1,
        e.BOMB_EXPLODED: 1,
        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 1,
        e.SURVIVED_ROUND: 10,
        e.GOT_KILLED: -10,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
