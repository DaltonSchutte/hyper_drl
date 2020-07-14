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
                 euclidean, lr, gamma, tau, update_freq, img, device):
        """
        Base class for a DQN based agent
            ARGS:
                batch_size(int):= number of experiences to sample from
                    the memory
                state_size(int):= dimension of the state
                hidden_dims(list):= list of layer widths (int) for the
                    hidden layers
                action_size(int):= number of available actions
                euclidean(bool):= True uses standard Euclidean geometry
                    False uses Hyperbolic embedding
                lr(float):= learning rate for network training
                gamma(0<=float<=1):= reward discount factor
                tau(float):= interpolation factor for updating the target
                    network with the online network
                    weights
                update_freq(int):= number of time steps between partial
                    updates to the target network
                img(bool):= True uses images as input and adds a conv head
                    to the networks False uses environment state
                device:= "cpu" or "cuda" for training
        """
        self.batch_size = batch_size
        self.state_size = state_size
        self.hidden_dims = hidden_dims
        self.action_size = action_size

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_freq = update_freq

        self.device = device

        assert isinstance(euclidean, bool)
        assert isinstance(img, bool)

        self.qnet_online = None
        self.qnet_target = None
        self.optimizer = None
        self.memory = None

        self.t = 0

    def init_components(self, seed):
        """
        Initializes the networks and memory using the seed provided
        """
        if self.euclidean:
            self.qnet_online = QNetwork(self.state_size, self.hidden_dims, self.action_size, self.img).to(self.device)
            self.qnet_target = QNetwork(self.state_size, self.hidden_dims, self.action_size, self.img).to(self.device)
            self.optimizer = optim.Adam(self.qnet_online.parameters(), lr=self.lr)
        else:
            self.qnet_online = HyperbolicQNetwork(self.state_size, self.hidden_dims,
                                                  self.action_size, self.img).to(self.device)
            self.qnet_target = HyperbolicQNetwork(self.state_size, self.hidden_dims,
                                                  self.action_size, self.img).to(self.device)
            self.optimizer = geoptim.RiemannianAdam(self.qnet_online.parameters(), lr=self.lr)

        self.memory = ExpReplay(capacity=int(1e5), action_size=self.action_size, batch_size=self.batch_size,
                                device=self.device, seed=seed)

    def act(self, state, epsilon):
        """
        Chooses an action with respect to the epsilon greedy policy
            ARGS:
                state:= environment state
                epsilon:= probability of choosing the greedy action
            RETURNS:
                action to be executed in the environment
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.qnet_online.eval()
        with torch.no_grad():
            act_values = self.qnet_online(state)
        self.qnet_online.train()

        if random.random() > epsilon:
            return np.argmax(act_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        """
        Adds an experience to the memory and, if applicable, makes
        a learning update
            ARGS:
                state:=
                action:=
                reward:=
                next_state:=
                done:=
        """
        self.memory.add_experience(state, action, reward, next_state, done)

        self.t = (self.t + 1) % self.update_freq

        if self.t == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """
        Perform a learning update using a batch of experiences sampled from
        the Experience Replay Memory
            ARGS:
                experiences:=
        """
        states, actions, rewards, next_states, dones = experiences

        a_max = torch.argmax(self.qnet_online(next_states), 1).unsqueeze(1)
        q_targets = rewards + (self.gamma * (1-dones) * self.qnet_target(next_states).gather(1, a_max))
        q_expected = self.qnet_online(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnet_target, self.qnet_online)

    def soft_update(self, target_net, online_net):
        """
        Performs a partial update to the target network weights using
        the online network weights
            ARGS:
                target_net:=
                online_net:=
        """
        for target_param, online_param in zip(target_net.parameters(),
                                              online_net.parameters()):
            target_param.data.copy(self.tau * online_param.data +
                                   (1.0-self.tau) * target_param.data)
