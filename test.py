import pytest

import numpy as np

import torch
from torch.nn import Sequential

from hdrl import agents
import hdrl.experiment as experiment
import hdrl.models as models

""" GLOBALS """
SEED = 314

STATE = np.array([1.1, 2.2, 3.3, 4.4])

# Euclidean Agent Variables and Setup
euclid_agent_params = {"batch_size": 32,
                       "state_size": 4,
                       "hidden_dims": [10,5],
                       "action_size": 2,
                       "euclidean": True,
                       "lr": 0.01,
                       "gamma": 0.999,
                       "tau": 0.001,
                       "update_freq": 4,
                       "img": False,
                       "device": 'cpu'}

euclidean_agent = agents.Agent(**euclid_agent_params)
euclidean_agent.init_components(SEED)

# Hyperbolic Agent Variables and Setup


# Unit Tests

class TestEuclideanAgent:
    # Tests for the online Q network
    def test_online_net_state_dim(self):
        assert euclidean_agent.qnet_online.state_dim == 4

    def test_online_net_hidden_dims(self):
        assert euclidean_agent.qnet_online.hidden_dims == [10, 5]

    def test_online_net_action_dim(self):
        assert euclidean_agent.qnet_online.action_dim == 2

    def test_online_net_body(self):
        assert isinstance(euclidean_agent.qnet_online.body, Sequential) == True

    # Tests for the target Q network
    def test_target_net_state_dim(self):
        assert euclidean_agent.qnet_target.state_dim == 4

    def test_target_net_hidden_dims(self):
        assert euclidean_agent.qnet_target.hidden_dims == [10, 5]

    def test_target_net_action_dim(self):
        assert euclidean_agent.qnet_target.action_dim == 2

    def test_target_net_body(self):
        assert isinstance(euclidean_agent.qnet_target.body, Sequential) == True

    # Memory tests
    def test_memory_batch_size(self):
        assert euclidean_agent.memory.batch_size == 32

    def test_memory_action_size(self):
        assert euclidean_agent.memory.action_size == 2

    def test_memory_device(self):
        assert euclidean_agent.memory.device == 'cpu'

    # Agent action test
    def test_act_greedy_policy(self):
        assert euclidean_agent.act(STATE, 0.0) in [0, 1]

    def test_act_random_policy(self):
        assert euclidean_agent.act(STATE, 1.0) in [0, 1]

