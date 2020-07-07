import random
from collections import deque, namedtuple

import numpy as np

import torch
import torch.optim as optim

import geoopt.optim as geoptim

from models import QNetwork, HyperbolicQNetwork


class ExpReplay(object):
    """ Experience Replay object """
    def __init__(self, capacity, action_size, batch_size, device, seed):
        """
        Creates an instance of the memory with capcity
            ARGS:
                capacity(int):= max number of experience
                    tuples to store in memory
        """
        self.n = capacity
        self.memory = deque(maxlen=capacity)
        self.action_size = action_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state",
                                                                "action",
                                                                "reward",
                                                                "next_state",
                                                                "done"]
                                     )
        self.device = device
        self.seed = random.seed(seed)

    # noinspection SpellCheckingInspection
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def add_experience(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def __len__(self):
        return len(self.memory)


class Agent:
    """ Agent to interact with and learn from the environment """
    def __init__(self, batch_size, state_size, hidden_dims, action_size,
                 euclidean, lr, gamma, tau, update_freq, seed, img, device):
        """
        Base class for a DQN based agent
            ARGS:
                batch_size(int):=
                state_size(int):=
                hidden_dims(list):=
                action_size(int):=
                euclidean(bool):=
                lr(float):=
                gamma(0<=float<=1):=
                tau(float):=
                update_freq(int):=
                seed(int):=
                img(bool):=
                device:=
        """
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.tau = tau
        self.update_freq = update_freq
        self.seed = seed

        self.device = device

        assert isinstance(euclidean, bool)
        assert isinstance(img, bool)
        if euclidean:
            self.qnet_online = QNetwork(state_size, hidden_dims, action_size, img).to(device)
            self.qnet_target = QNetwork(state_size, hidden_dims, action_size, img).to(device)
            self.optimizer = optim.Adam(self.qnet_online.parameters(), lr=lr)
        else:
            self.qnet_online = HyperbolicQNetwork(state_size, hidden_dims, action_size, img).to(device)
            self.qnet_target = HyperbolicQNetwork(state_size, hidden_dims, action_size, img).to(device)
            self.optimzier = geoptim.RiemannianAdam(self.qnet_online.parameters(), lr=lr)

        self.memory = ExpReplay(capacity=1e5, action_size=action_size, batch_size=batch_size,
                                device=device, seed=seed)

        self.t = 0

    def act(self, state, epsilon):
        """
        Chooses an action with respect to the epsilon greedy policy
            ARGS:
                state:= environment state
                epsilon:= probability of choosing the greedy action
            RETURNS:
                action to be executed in the environment
        """



