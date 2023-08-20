import numpy as np
import pickle
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS
from collections import namedtuple, deque
from .ppo import PPO2, HYPER, Transition, Memory
import torch
import torch.optim as optim
import torch.nn.functional as F

import torch.nn as nn
import settings

import wandb
# This is only an example!
#Transition = namedtuple('Transition',
#                        ('state', 'action', 'next_state', 'reward', 'prob_a', 'done'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"





def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.memory = Memory(10000)
    
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    self.optimizer = optim.Adam(self.model.parameters(), lr=HYPER.learning_rate)
    self.model.to(self.device)
    self.model.score = 0
    self.reward_history = []
    self.loss_history = []
    self.events_history = []
    self.logger.info(f'Set up model on device: {self.device}')
    if HYPER.WANDB:
        wandb.init(project="bomberman", name="bomberman-PPO",
                   config = HYPER._asdict(),
                    save_code=True,
                    ) 
    
    
    self.obs = []
    self.entropies = []
    self.values = []
    self.log_probs = []
    self.rewards = []
    self.dones = []
    self.actions = []
   
    self.loss_history = []
    
    self.global_step = 0
    self.n_games_played = 0


def one_hot(idx, num_actions=HYPER.N_ACTIONS):
    return idx
    one_hot = np.zeros(num_actions)
    one_hot[idx] = 1
    return one_hot

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
    
    
    #feature_state_old = state_to_features(self, old_game_state)
    #feature_state_new = state_to_features(self, new_game_state)
    if e.GOT_KILLED in events:
        done = True
    else:
        done = False
    action = ACTIONS.index(self_action)
    
    reward = reward_from_events(self, events)
    obs = self._obs
    logprob = self._log_prob
    value = self._value
    entropy = self._entropies
    #
    self.obs.append(obs)
    self.entropies.append(entropy)
    self.values.append(value)
    self.log_probs.append(logprob)
    self.rewards.append(reward)
    self.dones.append(done)
    self.actions.append(one_hot(action))

    #self.memory.push(obs,action,value,reward,logprob,entropy,done)
       
    #self.model.put_data((feature_state_old, action, reward / 100.0, feature_state_new, prob_a, done))
    #self.memory.push(feature_state_old, action, reward / 100.0, feature_state_new, prob_a, done)
    #self.memory.push(feature_state_old, action, 0, feature_state_new, 0, done)
    self.model.score += reward
    #self.logger.info(f'Score: {self.model.score}')
    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    # Store events for logging
    self.events_history.append(events)

    # Bootstrao values
    next_obs = state_to_features(self, new_game_state)
    next_obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)
    with torch.no_grad():
        next_value = self.model.get_value(next_obs).reshape(1,-1)
        # Convert tensor TODO: this is slow
        #self.rewards = torch.tensor(self.rewards, dtype=torch.float).to(self.device)
        #self.dones = torch.tensor(self.dones, dtype=torch.float).to(self.device)
        #self.values = torch.tensor(self.values, dtype=torch.float).to(self.device)
        if HYPER.GAE:
            self.advantages = [0 for _ in range(len(self.rewards))]
            lastgaelam = 0
        # TODO: Check if this is correct, what is Num_steps?
            steps = len(self.rewards)
            for t in reversed(range(steps)):
                #self.logger.info(f'Bootstrapping: t: {t}')
                if t == steps- 1:
                    next_non_terminal = 1.0 - done
                    nextvalues = next_value
                else:
                    next_non_terminal = 1.0 - self.dones[t+1]
                    nextvalues = self.values[t+1]
                delta = self.rewards[t] + HYPER.gamma * nextvalues * next_non_terminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + HYPER.gamma * HYPER.lmbda * next_non_terminal * lastgaelam
            self.returns = self.advantages + self.values
        else:
            raise NotImplementedError #TODO: Implement this
    


def optimize(self):
    self.logger.info(f'Optimizing...')  
    b_obs = torch.tensor(self.obs, dtype=torch.float).to(self.device)
    b_logprobs = torch.tensor(self.log_probs, dtype=torch.float).to(self.device)
    b_values = torch.tensor(self.values, dtype=torch.float).to(self.device)
    b_returns = torch.tensor(self.returns, dtype=torch.float).to(self.device)
    b_advantages = torch.tensor(self.advantages, dtype=torch.float).to(self.device)
    b_dones = torch.tensor(self.dones, dtype=torch.float).to(self.device)
    
    b_actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
    b_actions = b_actions#.reshape(-1,HYPER.N_ACTIONS)
    
    #Optimize policy for K epochs:
    inds = np.arange(HYPER.BATCH_SIZE)
    clipfracs = []
    if len(self.obs) < HYPER.BATCH_SIZE:
        self.logger.info(f'Not enough samples to train. Skipping...')
        return
    for i in range(HYPER.K_epoch):
        np.random.shuffle(inds)
        #TODO was ist wenn nicht genügend samples da sind?
        for start in range(0, HYPER.BATCH_SIZE, HYPER.MINI_BATCH_SIZE):
            end = start + HYPER.MINI_BATCH_SIZE
            minibatch_ind = inds[start:end]
       
            # convert one hot to int: TODO CHECK IF THIS IS TRUE
           # b_actions = torch.argmax(b_actions, dim=1)
            
            _,newlogprob,entropy,newvalue=self.model.get_action_and_value(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])
           
            
            
            logratio = newlogprob - b_logprobs[minibatch_ind]
            ratio = torch.exp(logratio)
            
            with torch.no_grad():
                # Calculate approximate kl divergence
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio-1)-logratio).mean()
                clipfracs +=[((ratio - 1.0).abs() > HYPER.CLIP_COEFF).float().mean().item()]

            mb_advantages = b_advantages[minibatch_ind]
            if HYPER.NORMALIZE_ADVANTAGE:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Calculate Policy Loss
            policy_loss1 = -ratio * mb_advantages
            policy_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - HYPER.CLIP_COEFF, 1.0 + HYPER.CLIP_COEFF)
            policy_loss = torch.max(policy_loss1, policy_loss2).mean()
            
            # Calculate Value Loss
            newvalue = newvalue.reshape(-1)
            if HYPER.CLIP_VALUE_LOSS:
                value_loss_unclipped = (newvalue - b_returns[minibatch_ind])**2
                value_clipped = b_values[minibatch_ind] + torch.clamp(newvalue - b_values[minibatch_ind], -HYPER.CLIP_COEFF, HYPER.CLIP_COEFF)
                value_loss_clipped = (value_clipped - b_returns[minibatch_ind])**2
                value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
                value_loss = 0.5 * value_loss_max.mean()
            else:
                value_loss = 0.5 * (newvalue - b_returns[minibatch_ind])**2
            
            entropy_loss = entropy.mean()
            loss = policy_loss - HYPER.ENTROPY_LOSS_COEFF * entropy_loss + HYPER.VALUE_LOSS_COEFF * value_loss
            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), HYPER.MAX_GRAD_NORM)
            self.optimizer.step()
            wandb.log({"loss": loss.item()})
            wandb.log({"policy_loss": policy_loss.item()})
            wandb.log({"value_loss": value_loss.item()})
            wandb.log({"entropy_loss": entropy_loss.item()})
            wandb.log({"approx_kl": approx_kl.item()})
            wandb.log({"value": b_values[minibatch_ind].mean().item()})
            wandb.log({"value_pred": newvalue.mean().item()})

            
            self.loss_history.append(loss.item())
            #TODO: maybe implement early stopping with KL divergence
            #if approx_kl > 1.5 * HYPER.target_kl:
            #    break
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
   # self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    # Train the model
    #self.logger.info(f'Starting to train...')
    
    self.n_games_played += 1
    
    
    # Optimize the model every 5 games
    if self.n_games_played % 20 == 0:
        optimize(self)
        self.logger.info(f'Training finished')
        
        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
        
        # Log important parameters to wandb TODO add more
        wandb.log({"cumulative_reward": self.model.score})
        wandb.log({"mean_reward": np.mean(self.reward_history)})
        wandb.log({"mean_loss": np.mean(self.loss_history)})
        self.logger.info(f'End of round: Reward_history: {self.reward_history}')
        #self.logger.info(f'End of round: event_history: {self.events_history}')
    # self.logger.info(f'End of round: action_histor: {[ACTIONS[a] for a in self.actions]}')
        # Survived n steps
        self.logger.info(f'End of round: Survived {last_game_state["step"]} steps')
        wandb.log({"survived steps": last_game_state["step"]})

        self.model.score = 0
        self.reward_history = []
        self.loss_history = []
        self.events_history = []




def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
   

    """_summary_

   
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
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    
    # Give reward depending on how long the agent survived
    
    
    self.reward_history.append(reward_sum)
    return reward_sum*0.1
   
    """
    game_rewards = {
        e.COIN_COLLECTED: 200,
        e.KILLED_OPPONENT: 600,
        e.KILLED_SELF: -400,
        # e.BOMB_DROPPED: 10,
        # e.COIN_FOUND: 1,
        # e.CRATE_DESTROYED: 10,
        e.INVALID_ACTION: -50,
        # e.SURVIVED_ROUND: 100,
        BOMB_DESTROYS_CRATE: 40,
        ESCAPABLE_BOMB: 75,
        # e.MOVED_LEFT: 1,
        # e.MOVED_DOWN: 1,
        # e.MOVED_RIGHT: 1,
        # e.MOVED_UP: 1,
        # e.WAITED: 0,
        WAITED_TOO_LONG: -350,
        CLOSER_TO_COIN: 40,
        IN_BLAST_RADIUS: -7,
        # e.BOMB_DROPPED: -1,
        FURTHER_FROM_COIN: -30,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    # Useless bomb

    if e.BOMB_DROPPED in events and BOMB_DESTROYS_CRATE not in events:
        reward_sum -= 600
        # print("useless")
    # Inescapable bomb
    if e.BOMB_DROPPED in events and ESCAPABLE_BOMB not in events:
        reward_sum += game_rewards[e.KILLED_SELF]
    self.reward_history.append(reward_sum)
    return reward_sum*0.01



#-------------------------------- TODO

CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"
ESCAPABLE_BOMB = "ESCAPABLE_BOMB"
BOMB_DESTROYS_CRATE = "BOMB_DESTROYS_CRATE"
WAITED_TOO_LONG = "WAITED_TOO_LONG"
IN_BLAST_RADIUS = "IN_BLAST_RADIUS"

from .callbacks import get_blast_coords

def reward_coin_distance(old_game_state, new_game_state, events):
    if old_game_state is None or new_game_state is None:
        return

    old_pos = np.array(old_game_state["self"][-1])
    new_pos = np.array(new_game_state["self"][-1])
    old_coins = np.array(old_game_state["coins"])
    new_coins = np.array(new_game_state["coins"])

    persistent_coins = old_coins[np.all(np.isin(old_coins, new_coins))]

    if persistent_coins.size == 0:
        return
    else:
        (persistent_coins,) = persistent_coins
    # Calculate old distances to all persistent coins
    old_distances = np.sqrt(np.sum((persistent_coins - old_pos) ** 2, axis=-1))
    # Find the index of the previous closest coin
    closest_coin_index = np.argmin(old_distances)
    # Get the distance to this previously closest coin now
    new_distance = np.sqrt(
        np.sum(((persistent_coins[closest_coin_index] - new_pos) ** 2))
    )
    # Also get the old distance.
    # Should be the minimum of `old_distances`.
    old_distance = old_distances[closest_coin_index]
    if new_distance > old_distance:
        events.append(FURTHER_FROM_COIN)

    elif new_distance < old_distance:
        events.append(CLOSER_TO_COIN)


def punish_long_wait(self, events):
    max_wait = settings.EXPLOSION_TIMER
    if e.WAITED in events:
        self.waited_for += 1
    else:
        self.waited_for = 0
    if self.waited_for > max_wait:
        events.append(WAITED_TOO_LONG)


def check_placed_bomb(old_features, new_game_state, events):
    escapable_bomb = False
    destroy_crate = False
    field = new_game_state["field"]
    name, score, is_bomb_possible, (player_x, player_y) = new_game_state["self"]
    if "BOMB_DROPPED" in events:
        for bomb in new_game_state["bombs"]:
            (bomb_x, bomb_y), timer = bomb
            if [bomb_x, bomb_y] == [player_x, player_y]:
                if old_features is not None:
                    escapable_bomb = bool(old_features[0][4] > 0)
                blast_coord = get_blast_coords(bomb, field)
                for coord in blast_coord:
                    if field[coord] == 1:
                        destroy_crate = True
                        return destroy_crate, escapable_bomb
    return destroy_crate, escapable_bomb


def check_blast_radius(game_state, events):
    fields = game_state["field"]
    _, _, _, (player_x, player_y) = game_state["self"]
    for bomb in game_state["bombs"]:
        blast_coord = get_blast_coords(bomb, fields)
        if (player_x, player_y) in blast_coord:
            events.append(IN_BLAST_RADIUS)
            return