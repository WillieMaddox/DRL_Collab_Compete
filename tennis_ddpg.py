import random
import copy
import time
from math import sqrt
from itertools import islice
from collections import deque
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from utils import transpose_list_to_list
from utils import transpose_to_tensor
from utils import policy_update
from unityagents import UnityEnvironment

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_cuda = torch.cuda.is_available()

device = "cpu"
use_cuda = device != "cpu"

OBSNORM = 1.0 / np.array([13, 7, 30, 7, 13, 7, 30, 7])


class UnityTennisEnv:
    """Unity Environment Wrapper

    """

    def __init__(self,
                 file_name='Tennis_Linux/Tennis.x86_64',
                 no_graphics=True,
                 normalize=False,
                 remove_ball_velocity=True):

        self.normalize = normalize
        self.remove_ball_velocity = remove_ball_velocity

        self.env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        self.num_agents = env_info.vector_observations.shape[0]
        self.state_size = self._get_obs(env_info.vector_observations).shape[1]
        self.action_size = brain.vector_action_space_size
        self.session_name = str(int(time.time()))

    def _get_obs(self, states):
        """Create obs from states"""
        states = states.reshape((self.num_agents, 3, 8))  # -> (n_agents, n_timesteps, n_obs)
        if self.normalize:
            states = states * OBSNORM[None, None, :]  # Normalize
        if self.remove_ball_velocity:
            states = states[:, :, :-2]  # BUG: temp fix to remove buggy ball velocity.
        return states.reshape((self.num_agents, -1))

    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        obs = self._get_obs(env_info.vector_observations)
        return obs.reshape(-1)

    def step(self, actions):
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = self.env.step(actions.reshape(self.num_agents, -1))[self.brain_name]
        obs_next = self._get_obs(env_info.vector_observations)
        rewards = np.array(env_info.rewards)
        dones = np.array(env_info.local_done).astype(np.float)
        return obs_next.reshape(-1), rewards, dones

    def close(self):
        self.env.close()


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size,
                 buffer_size=int(1e5),
                 batch_size=256,
                 learn_every=1,
                 update_every=1,
                 gamma=0.99,
                 tau=0.02,
                 lr_actor=2e-4,
                 lr_critic=2e-3,
                 random_seed=None,
                 use_asn=True,
                 asn_kwargs={},
                 use_psn=False,
                 psn_kwargs={},
                 restore=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.update_every = update_every
        self.learn_every = learn_every
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        # Keep track of how many times we've updated weights
        self.i_updates = 0
        self.i_step = 0
        self.use_asn = use_asn
        self.use_psn = use_psn

        if random_seed is not None:
            random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        if self.use_psn:
            self.actor_perturbed = Actor(state_size, action_size).to(device)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)

        # restore networks if needed
        if restore is not None:
            checkpoint = torch.load(restore, map_location=device)
            self.actor_local.load_state_dict(checkpoint[0]['actor'])
            self.actor_target.load_state_dict(checkpoint[0]['actor'])
            if self.use_psn:
                self.actor_perturbed.load_state_dict(checkpoint[0]['actor'])
            self.critic_local.load_state_dict(checkpoint[0]['critic'])
            self.critic_target.load_state_dict(checkpoint[0]['critic'])

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)

        # Hard copy weights from local to target networks
        policy_update(self.actor_local, self.actor_target, 1.0)
        policy_update(self.critic_local, self.critic_target, 1.0)

        # Noise process
        if self.use_asn:
            self.action_noise = OUNoise(action_size, **asn_kwargs)

        if self.use_psn:
            self.param_noise = PSNoise(**psn_kwargs)

        self.buffer = ExperienceReplay(buffer_size, batch_size)

    def act(self, states, perturb_mode=True, train_mode=True):
        """Returns actions for given state as per current policy."""
        if not train_mode:
            self.actor_local.eval()
            if self.use_psn:
                self.actor_perturbed.eval()

        with torch.no_grad():
            states = torch.from_numpy(states).float().to(device)
            actor = self.actor_perturbed if (self.use_psn and perturb_mode) else self.actor_local
            actions = actor(states).cpu().numpy()[0]

        if train_mode:
            actions += self.action_noise.sample()

        self.actor_local.train()
        if self.use_psn:
            self.actor_perturbed.train()

        return np.clip(actions, -1, 1)

    def perturb_actor_parameters(self):
        """Apply parameter space noise to actor model, for exploration"""
        policy_update(self.actor_local, self.actor_perturbed, 1.0)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            random = torch.randn(param.shape)
            if use_cuda:
                random = random.cuda()
            param += random * self.param_noise.current_stddev

    def reset(self):
        self.action_noise.reset()
        if self.use_psn:
            self.perturb_actor_parameters()

    def step(self, experience):
        self.buffer.push(experience)
        self.i_step += 1
        if len(self.buffer) > self.batch_size:
            if self.i_step % self.learn_every == 0:
                self.learn()
            if self.i_step % self.update_every == 0:
                self.update()  # soft update the target network towards the actual networks

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
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # ---------------------------- update critic ---------------------------- #
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_local.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_local.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update(self):
        """soft update targets"""
        self.i_updates += 1
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

    def postprocess(self, t_step):
        if self.use_psn and t_step > 0:
            perturbed_states, perturbed_actions, _, _, _ = self.buffer.tail(t_step)
            unperturbed_actions = self.act(np.array(perturbed_states), False, False)
            diff = np.array(perturbed_actions) - unperturbed_actions
            mean_diff = np.mean(np.square(diff), axis=0)
            dist = sqrt(np.mean(mean_diff))
            self.param_noise.adapt(dist)


class PSNoise:
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
        self.deque.append(transition)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        samples = random.sample(self.memory, k=self.batch_size)
        return transpose_to_tensor(samples)

    def tail(self, n):
        samples = list(islice(self.memory, len(self.memory) - n, len(self.memory)))
        return transpose_list_to_list(samples)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


