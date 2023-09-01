


import torch
import torch.nn as nn
import torch.optim as optim



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class HYPER:
    kernel_size = 3
    stride = 1
    model_name = "Qcnn.pt"
    learning_rate = 0.0001
    gamma = 0.99
    batch_size = 32
    memory_size = 10000
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 1000
    target_update = 1
    input_channels = 6
    activation_function = nn.Tanh()
    tau = 0.005






class Qcnn(nn.Module):
    def __init__(self, input_channels=6, num_actions=len(ACTIONS)):
        super(Qcnn, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=HYPER.kernel_size, stride=HYPER.stride)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization after first conv layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization after second conv layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256)  # Batch normalization after third conv layer
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)  # Added an extra convolutional layer
        self.bn4 = nn.BatchNorm2d(256)  # Batch normalization after fourth conv layer
        
        # Adjust the dimensions for the fully connected layer based on the output dimensions from the conv layers
        self.fc1 = nn.Linear(1024, 1024)  # Adjust the input size here based on the size after the final conv layer
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer for regularization
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)  # Another dropout layer
        self.fc3 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = HYPER.activation_function(self.bn1(self.conv1(x)))
        x = HYPER.activation_function(self.bn2(self.conv2(x)))
        x = HYPER.activation_function(self.bn3(self.conv3(x)))
        x = HYPER.activation_function(self.bn4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = HYPER.activation_function(self.fc1(x))
        x = self.dropout1(x)
        x = HYPER.activation_function(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


# class Qcnn(nn.Module):
#     def __init__(self, input_channels=6, num_actions= len(ACTIONS)):
#         super(Qcnn, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=HYPER.kernel_size, stride=HYPER.stride)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         # Adjust this linear layer's input dimensions based on the output size after conv3.
#         # This is a naive computation; it may need adjustments.
#         self.fc1 = nn.Linear(1024, 512)  
#         self.fc2 = nn.Linear(512, num_actions)

#     def forward(self, x):
#         x = HYPER.activation_function(self.conv1(x))
#         x = HYPER.activation_function(self.conv2(x))
#         x = HYPER.activation_function(self.conv3(x))
#         x = x.view(x.size(0), -1)  # flatten
#         x = HYPER.activation_function(self.fc1(x))
#         return self.fc2(x)





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

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate,amsgrad=True)
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
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item(), state_action_values.mean().item(), expected_state_action_values.mean().item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
