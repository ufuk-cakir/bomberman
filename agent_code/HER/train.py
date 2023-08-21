from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features 

from .Qmodel import QNet, Memory, Transition,HYPER, ACTIONS
import torch
import torch.optim as optim
import torch.nn.functional as F



#----------
import wandb


# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

self.opt = Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.001
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
    wandb.init(project="bomberman-qagent", name="training-run-1")
    wandb.config.update(HYPER._asdict())
    self.reward_history = []
    self.loss_history = []
    self.events_history = []
    self.avtion_history = []
    




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
    
   # self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # Idea: Add your own events to hand out rewards
    #if ...:
     #   events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    if len(self.memory) > HYPER.BATCH_SIZE:
        optimize_model(self)
        
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
    wandb.log({"loss": loss.item(), "cumulative_reward": self.score})
    wandb.watch(self.policy_net)
    
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
    wandb.log({"loss": loss.item(), "cumulative_reward": self.score})
    wandb.watch(self.policy_net)
    
    # Compute gradient statistics
    for name, param in self.policy_net.named_parameters():
        if param.grad is not None:
            wandb.log({f"gradient/{name}": param.grad.norm()})



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
    state = state_to_features(self,last_game_state)
    state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
    final_state = None
    action = ACTIONS.index(last_action)
    action = torch.tensor([[action]], device=self.device, dtype=torch.long)
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

    wandb.log({"final_cumulative_reward": self.score}) 
    self.score = 0
    self.memory = Memory(10000)
    
    
    
    # do also save on wandb
    
    # Log the events string using Weights and Biases
    # Idea: Store the q-values corresponding to your last action somewhere.
    #wandb.run.finish()
    wandb.save("logs/Qagent.log")
    wandb.save("policy_net.pt")
    wandb.save("target_net.pt") 
    
    wandb.log({"events happend": self.events_history})
    wandb.log({"actions taken by agent": self.avtion_history})


    # Calculate statistics for logging: TODO stuff like average reward, etc.
    

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 200,
        e.KILLED_OPPONENT: 600,
        e.KILLED_SELF: -600,  # idea: the custom event is bad
        e.INVALID_ACTION: -50,
        e.WAITED: -10,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.BOMB_DROPPED: 50,
        e.BOMB_EXPLODED: 15,
        e.CRATE_DESTROYED: 50,
        e.COIN_FOUND: 100,
        e.SURVIVED_ROUND: 500,
        e.GOT_KILLED: -50,
    }
    
    game_rewards = {
    e.COIN_COLLECTED: 2.0,
    e.KILLED_OPPONENT: 6.0,
    e.KILLED_SELF: -2.0,
    e.INVALID_ACTION: -0.5,
    e.WAITED: -0.1,
    e.MOVED_LEFT: 0.1,
    e.MOVED_RIGHT: 0.1,
    e.MOVED_UP: 0.1,
    e.MOVED_DOWN: 0.1,
    e.BOMB_DROPPED: 0.5,
    e.BOMB_EXPLODED: 0.15,
    e.CRATE_DESTROYED: 0.5,
    e.COIN_FOUND: 1.0,
    e.SURVIVED_ROUND: 5.0,
    e.GOT_KILLED: -2.0,
}
    reward_sum = 0 # default reward for surviving
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    wandb.log({"reward": reward_sum})
    return reward_sum

