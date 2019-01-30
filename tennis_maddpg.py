import random
import time
from collections import namedtuple
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from unityagents import UnityEnvironment
from model import Actor, Critic
from utils import policy_update
from tennis_ddpg import OUNoise, PSNoise
from tennis_ddpg import ExperienceReplay, PrioritizedExperienceReplay

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_cuda = torch.cuda.is_available()

device = "cpu"
use_cuda = device != "cpu"

OBSNORM = 1.0 / np.array([13, 7, 30, 7, 13, 7, 30, 7])

Obs = namedtuple("Obs", field_names=["px", "py", "vx", "vy", "bx", "by"])


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
            states = states[:, :, :-2]  # remove buggy ball velocity.
        return states.reshape((self.num_agents, -1))

    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        obs = self._get_obs(env_info.vector_observations)
        return obs

    def step(self, actions):
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = self.env.step(actions.reshape(self.num_agents, -1))[self.brain_name]
        obs_next = self._get_obs(env_info.vector_observations)
        rewards = np.array(env_info.rewards).reshape((2, 1))
        dones = np.array(env_info.local_done).reshape((2, 1)).astype(np.float)
        return obs_next, rewards, dones

    def close(self):
        self.env.close()


class Agent:
    """Interacts with and learns from the environment."""
    buffer = None
    share_buffer = False

    def __init__(self, state_size, action_size, i_agent,
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
                 use_per=False,
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
        self.i_agent = i_agent
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
        self.use_per = use_per

        if random_seed is not None:
            random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        if self.use_psn:
            self.actor_perturbed = Actor(state_size, action_size).to(device)
        self.critic_local = Critic(2 * state_size, action_size).to(device)
        self.critic_target = Critic(2 * state_size, action_size).to(device)

        # restore networks if needed
        if restore is not None:
            checkpoint = torch.load(restore, map_location=device)
            self.actor_local.load_state_dict(checkpoint[self.i_agent]['actor'])
            self.actor_target.load_state_dict(checkpoint[self.i_agent]['actor'])
            if self.use_psn:
                self.actor_perturbed.load_state_dict(checkpoint[self.i_agent]['actor'])
            self.critic_local.load_state_dict(checkpoint[self.i_agent]['critic'])
            self.critic_target.load_state_dict(checkpoint[self.i_agent]['critic'])

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

        if self.use_per:
            replay_class = PrioritizedExperienceReplay
        else:
            replay_class = ExperienceReplay

        # Replay memory
        if Agent.share_buffer:
            if Agent.buffer is None:
                Agent.buffer = replay_class(buffer_size, batch_size, random_seed)
            self.buffer = Agent.buffer
        else:
            self.buffer = replay_class(buffer_size, batch_size, random_seed)

    def act(self, states, perturb_mode=True, train_mode=True):
        """Returns actions for given state as per current policy."""
        if not train_mode:
            self.actor_local.eval()
            if self.use_psn:
                self.actor_perturbed.eval()

        with torch.no_grad():
            states = torch.from_numpy(states).float().to(device)
            actor = self.actor_perturbed if (self.use_psn and perturb_mode) else self.actor_local
            actions = actor(states).cpu().numpy()

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

    def step(self, experience, priority=0.0):
        self.buffer.push(experience)
        self.i_step += 1

        if len(self.buffer) > self.batch_size:
            if self.i_step % self.learn_every == 0:
                self.learn(priority)
            if self.i_step % self.update_every == 0:
                self.update()  # soft update the target network towards the actual networks

    def learn(self, priority):
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
        if self.use_per:
            (states, actions, rewards, states_next, dones), batch_idx = self.buffer.sample(priority)
        else:
            states, actions, rewards, states_next, dones = self.buffer.sample()

        states_actor, states_critic = states[:, :self.state_size], states[:, self.state_size:]
        states_actor_next, states_critic_next = states_next[:, :self.state_size], states_next[:, self.state_size:]

        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            actions_next = self.actor_target(states_actor_next)
            Q_targets_next = self.critic_target(states_critic_next, actions_next)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # ---------------------------- update critic ---------------------------- #
        # Compute critic loss
        Q_expected = self.critic_local(states_critic, actions)
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_local.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states_actor)
        actor_loss = -self.critic_local(states_critic, actions_pred).mean()

        # Minimize the loss
        self.actor_local.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.use_per:
            Q_error = Q_expected - Q_targets
            new_deltas = torch.abs(Q_error.detach().squeeze(1)).numpy()
            self.buffer.update_deltas(batch_idx, new_deltas)

    def update(self):
        """soft update targets"""
        self.i_updates += 1
        policy_update(self.actor_local, self.actor_target, self.tau)
        policy_update(self.critic_local, self.critic_target, self.tau)

    def get_save_dict(self):

        save_dict = {'actor': self.actor_local.state_dict(),
                     'actor_optim_params': self.actor_optimizer.state_dict(),
                     'critic': self.critic_local.state_dict(),
                     'critic_optim_params': self.critic_optimizer.state_dict()}
        return save_dict

    def postprocess(self, t_step, i_agent):
        if self.use_psn and t_step > 0:
            perturbed_states, perturbed_actions, _, _, _ = self.buffer.tail(t_step, i_agent + 1)
            states_actor = perturbed_states[:, :self.state_size]
            unperturbed_actions = self.act(np.array(states_actor), False, False)
            diff = np.array(perturbed_actions) - unperturbed_actions
            mean_diff = np.mean(np.square(diff), axis=0)
            dist = np.sqrt(np.mean(mean_diff))
            self.param_noise.adapt(dist)
