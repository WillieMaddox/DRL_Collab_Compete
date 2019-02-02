import os
import random
import copy
from itertools import islice
from collections import deque
from collections import namedtuple

import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable

Obs = namedtuple("Obs", field_names=["px", "py", "vx", "vy", "bx", "by"])


def transpose_list_to_list(mylist):
    return list(map(list, zip(*mylist)))


def transpose_array_to_list(mylist):
    def make_array(x):
        return np.array(x, dtype=np.float)
    return list(map(make_array, zip(*mylist)))


def transpose_to_tensor(input_list):
    def make_tensor(x):
        return torch.tensor(x, dtype=torch.float)
    return list(map(make_tensor, zip(*input_list)))


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def policy_update(local, target, tau):
    """
    Perform DDPG soft update (move target params toward local based on weight
    factor tau) With tau == 1, this function performs a hard_update.
    Inputs:
        local (torch.nn.Module): Net whose parameters to copy
        target (torch.nn.Module): Net to copy parameters to
        tau (float, 0 < x <= 1): Weight factor for update
    """
    for target_param, local_param in zip(target.parameters(), local.parameters()):
        target_param.data.copy_(local_param.data * tau + target_param.data * (1.0 - tau))


# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in enumerate(torch.rand(logits.shape[0]))])


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=0.5, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


class ParameterSpaceNoise:
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient
        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:  # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:  # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        return {'param_noise_stddev': self.current_stddev}

    def __repr__(self):
        fmt = 'PSNoise(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2, scale_start=1.0, scale_end=0.01, decay=1.0):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.scale = scale_start
        self.scale_end = scale_end
        self.decay = decay
        self.state = copy.copy(self.mu)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        s = self.scale
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.mu.shape)
        self.state = x + dx
        self.scale = max(self.scale * self.decay, self.scale_end)
        return self.state * s


class ExperienceReplay:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=None):
        """Initialize an Experience Replay Buffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.seed = random.seed(seed) if seed is not None else seed
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)

    def push(self, transition):
        """push onto the buffer"""
        self.memory.append(transition)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        samples = random.sample(self.memory, k=self.batch_size)
        return transpose_to_tensor(samples)

    def tail(self, n, offset=0):
        if offset == 0:
            samples = list(islice(self.memory, len(self.memory) - n, len(self.memory)))
            samples = transpose_list_to_list(samples)
        else:
            samples = [self.memory[i] for i in range((len(self.memory) - 2 * n) + (offset - 1), len(self.memory), 2)]
            samples = transpose_array_to_list(samples)
        return samples

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedExperienceReplay:
    def __init__(self, buffer_size, batch_size, seed=None, min_delta=1e-5):
        """Initialize a Prioritized Experience Replay Buffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.seed = random.seed(seed) if seed is not None else seed
        self.batch_size = batch_size
        self.min_delta = min_delta
        self.memory = deque(maxlen=buffer_size)
        self.deltas = deque(maxlen=buffer_size)

    def push(self, transition):
        """push onto the buffer"""
        self.memory.append(transition)
        self.deltas.append(max(self.deltas) if len(self.deltas) > 0 else self.min_delta)

    def sample(self, priority=0.5):
        deltas = np.array(self.deltas)
        probs = deltas ** priority / np.sum(deltas ** priority)
        exp_batch_idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs, replace=False)
        samples = [self.memory[idx] for idx in exp_batch_idx]
        return transpose_to_tensor(samples), exp_batch_idx

    def tail(self, n, offset=0):
        if offset == 0:
            samples = list(islice(self.memory, len(self.memory) - n, len(self.memory)))
            samples = transpose_list_to_list(samples)
        else:
            samples = [self.memory[i] for i in range((len(self.memory) - 2 * n) + (offset - 1), len(self.memory), 2)]
            samples = transpose_array_to_list(samples)
        return samples

    def update_deltas(self, idxs, deltas):
        for i, idx in enumerate(idxs):
            self.deltas[idx] = deltas[i] + self.min_delta

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


"""def main():
    torch.Tensor()
    print(onehot_from_logits())

if __name__=='__main__':
    main()"""
