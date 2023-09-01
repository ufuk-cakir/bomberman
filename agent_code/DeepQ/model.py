


import torch
import torch.nn as nn
import torch.optim as optim



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class HYPER:
    kernel_size = 3
    stride = 1
    model_name = "Qcnn.pt"
    learning_rate = 0.001
    gamma = 0.99
    batch_size = 32
    memory_size = 1000
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 200
    target_update = 10
    input_channels = 6
    






class Qcnn(nn.Module):
    def __init__(self, input_channels=6, num_actions= len(ACTIONS)):
        super(Qcnn, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=HYPER.kernel_size, stride=HYPER.stride)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Adjust this linear layer's input dimensions based on the output size after conv3.
        # This is a naive computation; it may need adjustments.
        self.fc1 = nn.Linear(1024, 512)  
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)





from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class deepQ:
    def __init__(self, input_channels, num_actions, device,learning_rate=HYPER.learning_rate, gamma=HYPER.gamma, batch_size=HYPER.batch_size, memory_size=HYPER.memory_size):
        self.policy_net = Qcnn(input_channels, num_actions).to(device)
        self.target_net = Qcnn(input_channels, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.device = device


    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
