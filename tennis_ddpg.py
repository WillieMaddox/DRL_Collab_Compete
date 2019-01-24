import random
import copy
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from utils import transpose_list, transpose_to_tensor
from utils import policy_update

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.02             # for soft update of target parameters
LR_ACTOR = 2e-4         # Learning rate of the actor
LR_CRITIC = 2e-3        # Learning rate of the critic
WEIGHT_DECAY = 0    # L2 weight decay
NOISE_DECAY = 0.99995   #
UPDATE_EVERY = 1        # Update the network after this many steps.
NUM_BATCHES = 1         # Roll out this many batches when training.
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

OBSNORM = 1.0 / np.array([13, 7, 30, 7, 13, 7, 30, 7])

Obs = namedtuple("Obs", field_names=["px", "py", "vx", "vy", "bx", "by"])


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, restore=None):
        """Initialize an Agent0 object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.tau = TAU
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        # self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # restore networks if needed
        if restore is not None:
            checkpoint = torch.load(restore, map_location=device)
            self.actor_local.load_state_dict(checkpoint[0]['actor'])
            self.actor_target.load_state_dict(checkpoint[0]['actor'])
            self.critic_local.load_state_dict(checkpoint[0]['critic'])
            self.critic_target.load_state_dict(checkpoint[0]['critic'])

        # Hard copy weights from local to target networks
        policy_update(self.actor_local, self.actor_target, 1.0)
        policy_update(self.critic_local, self.critic_target, 1.0)

        # Noise process
        # self.noise = OUNoise0(num_agents, action_size)
        self.noise = OUNoise(None, action_size)
        self.noise_scale = 1.0
        self.count = 0
        self.epsilon = 1.0

        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        # Keep track of how many times we've updated weights
        self.n_updates = 0
        self.n_steps = 0

    def get_obs(self, states):
        """Create obs and obs_full from states"""
        states = states.reshape((2, 3, 8))  # -> (n_agents, n_timesteps, n_obs)
        # states = states * OBSNORM[None, None, :]  # Normalize
        states = states[:, :, :-2]  # remove buggy ball velocity.
        obs = states.reshape((2, -1))
        return obs

    def act(self, states, train_mode=True):
        """Returns actions for given state as per current policy."""
        if not train_mode:
            self.actor_local.eval()

        with torch.no_grad():
            states = torch.from_numpy(states).float().to(device)
            actions = self.actor_local(states).cpu().numpy()
            # actions = self.actor_local(states).detach().numpy()

        if train_mode:
            actions += self.noise.sample() * self.noise_scale
            self.noise_scale = max(self.noise_scale * NOISE_DECAY, 0.01)

        self.actor_local.train()

        return np.clip(actions, -1, 1)

    def reset(self):
        self.count += 1
        self.epsilon = np.exp(-0.0005 * self.count)
        self.noise.reset()

    def step(self, experience):
        self.buffer.push(experience)
        self.n_steps += 1
        if self.n_steps % UPDATE_EVERY == 0 and self.n_steps > BATCH_SIZE * NUM_BATCHES:
            for _ in range(NUM_BATCHES):
                self.learn()
                self.update_targets()  # soft update the target network towards the actual networks

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, states_next, dones = self.buffer.sample()

        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            actions_next = self.actor_target(states_next)
            Q_targets_next = self.critic_target(states_next, actions_next)
            Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        # ---------------------------- update critic ---------------------------- #
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        # critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_local.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_local.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_targets(self):
        """soft update targets"""
        self.n_updates += 1
        policy_update(self.actor_local, self.actor_target, self.tau)
        policy_update(self.critic_local, self.critic_target, self.tau)

    def save_model(self, filename):
        save_dict_list = []
        save_dict = {'actor': self.actor_local.state_dict(),
                     'actor_optim_params': self.actor_optimizer.state_dict(),
                     'critic': self.critic_local.state_dict(),
                     'critic_optim_params': self.critic_optimizer.state_dict()}
        save_dict_list.append(save_dict)
        torch.save(save_dict_list, filename)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, n_agents, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        if n_agents is None:
            self.mu = mu * np.ones(size)
        else:
            self.mu = mu * np.ones((n_agents, size))
        self.theta = theta
        self.sigma = sigma
        self.state = copy.copy(self.mu)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.mu.shape)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.deque = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, transition):
        """push into the buffer"""
        input_to_buffer = transpose_list(transition)
        for item in input_to_buffer:
            self.deque.append(item)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        samples = random.sample(self.deque, k=self.batch_size)
        # transitions = transpose_list(samples)
        transitions = transpose_to_tensor(samples)
        return transitions

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.deque)
