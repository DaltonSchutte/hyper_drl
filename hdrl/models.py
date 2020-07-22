from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import geoopt
import geoopt.manifolds as M


class QNetwork(nn.Module):
    """Euclidean version of the Q-Network"""
    def __init__(self, state_dim, hidden_dims, action_dim, img=False):
        super().__init__()
        """"
        Creates an instance of a Q-Network with specified parameters. Can
        be configured to have a convolutional head for use with atari image
        input
            PARAMS:
                state_dim(int):= dimension of the input state
                hidden_dim(list of ints):= dimension for each hidden layer
                action_dim(int):= dimension of the output
                img(bool):= indicates if image input will be used
        """
        self.state_dim = state_dim

        assert type(hidden_dims) == list
        assert len(hidden_dims) > 0

        self.hidden_dims = hidden_dims
        self.action_dim = action_dim

        if img:
            self.head = self.build_conv_head()

        self.body = OrderedDict()

        for i, dim in enumerate(self.hidden_dims):
            if i == 0:
                self.body.update({'in_layer': nn.Linear(self.state_dim, dim)})
                self.body.update({f'relu_{i}': nn.ReLU()})
            else:
                self.body.update({f'layer_{i}': nn.Linear(self.hidden_dims[i-1], dim)})
                self.body.update({f'relu_{i}': nn.ReLU()})

        self.body.update({'out_layer': nn.Linear(self.hidden_dims[-1], self.action_dim)})

        self.body = nn.Sequential(self.body)

    def build_conv_head(self):
        """
        Builds a series of convolutional layers per specifications
        NOT YET IMPLEMENTED
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Conducts a forward pass through the network to produce a Q-value
        for a given state
            ARGS:
                x:= state
            RETURNS:
                Q-value for the given state
        """
        q = self.body(x)
        return q


class HyperbolicQNetwork(nn.Module):
    """ Hyperbolic Version of the Q-Network"""
    def __init__(self, state_dim, hidden_dims, action_dim, img=False):
        """
        Creates an instance of a Q-Network with specified parameters using
        hyperbolic network components. Can be configured to have a conv
        head for use with atari image
        input
            PARAMS:
                state_dim(int):= dimension of the input state
                hidden_dim(list of ints):= dimension for each hidden layer
                action_dim(int):= dimension of the output
                img(bool):= indicates if image input will be used
        """
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.img = img

    def build_conv_head(self):
        """
        Builds a series of convolutional layers per specifications
        NOT YET IMPLEMENTED
        """
        raise NotImplementedError

    def forward(self, x):
        pass


def create_ball(ball=None, c=None):
    """

    """
    if ball is None:
        assert c is not None, "curvature of ball should be defined"
        ball = geoopt.PoincareBall(c)
    return ball


def mobius_linear(x, weight, bias=None, nonlin=None, *, ball: geoopt.PoincareBall):
    """

    """
    output = ball.mobius_matvec(weight, x)
    if bias is not None:
        output = ball.mobius_add(output, bias)
    if nonlin is not None:
        output = ball.logmap0(output)
        output = nonlin(output)
        output = ball.expmap0(output)
    return output


class MobiusLinear(nn.Linear):
    def __init__(self, *args, nonlin=None, ball=None, c=1.0, **kwargs):
        """
        Implementation of the Mobius Linear Layer as described in  Hyperbolic
        Neural Networks Ganea et al. 2018. Heavily borrowed from
        https://github.com/geoopt/geoopt/blob/master/examples/mobius_linear_example.py
            PARAMS:
                nonlin:=
                ball:=
                c:=
        """
        super().__init__(*args, **kwargs)
        self.ball = create_ball(ball, c)
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, x):
        out = mobius_linear(x,
                            weight=self.weight,
                            bias=self.bias,
                            nonlin=self.nonlin,
                            ball=self.ball
                            )
        return out

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
        if self.bias is not None:
            self.bias.zero_()
