import pytest

import numpy as np

from torch.nn import Sequential

from . import agents, experiment, models

""" GLOBALS """
SEED = 314

STATE = np.array([1, 2, 3, 4])

# Euclidean Agent Variables and Setup
euclid_agent_params = {"batch_size": 32,
                       "state_size": 4,
                       "hidden_dims": [10],
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
    def test_online_net(self):
        assert euclidean_agent.qnet_online.state_dim == 4
        assert euclidean_agent.qnet_online.hidden_dims == [10]
        assert euclidean_agent.qnet_online.action_dim == 2
        assert isinstance(euclidean_agent.qnet_online.body, Sequential) ==  True

    def test_target_net(self):
        assert euclidean_agent.qnet_target.state_dim == 4
        assert euclidean_agent.qnet_target.hidden_dims == [10]
        assert euclidean_agent.qnet_target.action_dim == 2
        assert isinstance(euclidean_agent.qnet_target.body, Sequential) == True

    def test_memory(self):
        assert euclidean_agent.memory.batch_size == 32
        assert euclidean_agent.memory.action_size == 2
        assert euclidean_agent.memory.device == 'cpu'

    def test_act(self):
        assert euclidean_agent.act(STATE, 0.05) in [0,1]
