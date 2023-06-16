import os
import sys
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import flatten
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_actions = 8
update_interval = 50
batch_size = 32
memory_size = 50000
learning_rate = 0.001
gamma = 0.65
input_channel = 2

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(2, 4), stride=(1, 2))  # output shape (32, 49, 7)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 2))  # output shape (32, 48, 3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1))  # output shape (32, 47, 2)

        self.fc1 = nn.Linear(32*47*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN:
    def __init__(self, model_path=None):
        # Initialize attributes
        # self._input_dim = input_dim
        self._action_size = n_actions
        self.model_path = model_path
        self.learn_step = 0
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.memory_size = memory_size

        # Initialize discount and exploration rate
        self.gamma = gamma
        self.learning_rate = learning_rate

        # Initialize epsilon parameters
        self.max_epsilon = 0.9
        self.min_epsilon = 0.01
        self.epsilon_decay = -0.1

        # Is training, initialize memory and build model
        if not self.model_path:
            self.experience_replay = deque([], maxlen=self.memory_size)
            self.q_network, self.target_network = Net().to(device), Net().to(device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            self.loos_func = nn.MSELoss()

        # Is testing, load model
        elif self.model_path:
            self.q_network = self.load_model()

    def save_model(self, model_path):
        print('Saving model')
        torch.save(self.q_network, os.path.join(model_path, 'training_model.pth'))

    def load_model(self):
        model_path = os.path.join(self.model_path, 'training_model.pth')
        if os.path.isfile(model_path):
            print('Model found')
            return torch.load(model_path)
        else:
            sys.exit('Model not found')

    def store(self, state, action, reward, next_state):
        self.experience_replay.append(Transition(state, action, reward, next_state))
        # print('experience_replay size:', len(self.experience_replay))

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            # print('---random draw---')
            return torch.tensor([[random.randint(0, self._action_size - 1)]], dtype=torch.long, device=device)
        else:
            with torch.no_grad():
                # print(self.q_network(state).max(1))
                return self.q_network(state).max(1)[1].view(1, 1)

    def get_epsilon(self, episode):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(self.epsilon_decay * episode)

    # @profile
    def train(self):

        # Update the target network
        if self.learn_step % self.update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            # print(self.learn_step, 'update target network')
        self.learn_step += 1

        # Sample batch memory from all experiences
        # print('memory size:', len(self.experience_replay), sys.getsizeof(self.experience_replay))
        if len(self.experience_replay) > self.batch_size:
            transitions = random.sample(self.experience_replay, self.batch_size)
        else:
            transitions = self.experience_replay

        batch = Transition(*zip(*transitions))
        # print(type(batch.action), type(batch.action[0]), type(batch.state), type(batch.state[0]))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = self.q_network(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_state_values = self.target_network(next_state_batch).max(1)[0]

        expect_state_action_values = reward_batch + (self.gamma * next_state_values)

        loss = self.loos_func(state_action_values, expect_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print(self.learn_step)
