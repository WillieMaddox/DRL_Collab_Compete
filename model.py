import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(Actor, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.nonlin = F.relu
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = self.fc1(state)
        x = self.nonlin(x)
        x = self.fc2(x)
        x = self.nonlin(x)
        x = self.fc3(x)
        action = F.tanh(x)
        action = torch.clamp(action, -1, 1)
        return action


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.nonlin = F.relu
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = self.fc1(state)
        x = self.nonlin(x)
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = self.nonlin(x)
        q = self.fc3(x)
        # q = torch.clamp(q, -1, 1)
        return q


class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.nonlin = F.relu  # leaky_relu
        self.actor = actor
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        # if x.dim() == 1:
        #     x = torch.unsqueeze(x, 0)
        h1 = self.nonlin(self.fc1(x))
        # h1 = self.bn1(h1)
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.fc3(h2)

        if self.actor:
            # return a vector of the force

            # h3 is a 2D vector (a force that is applied to the agent)
            # we bound the norm of the vector to be between 0 and 10
            # norm = torch.norm(h3)
            # hout = 10.0 * (F.tanh(norm)) * h3 / norm if norm > 0 else 10 * h3

            out = F.tanh(h3)
            return out
        else:
            out = torch.clamp(h3, -1, 1)
            return out
