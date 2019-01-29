import random
import copy
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic, Network
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


class MADDPG:
    def __init__(self, state_size, action_size, num_agents):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 24+2+2=28
        in_actor = 18
        in_critic = 36
        self.agents = [DDPGAgent(in_actor, 32, 16, action_size, in_critic + action_size * num_agents, 32, 16),
                       DDPGAgent(in_actor, 32, 16, action_size, in_critic + action_size * num_agents, 32, 16)]

        self.epsilon = 1.0
        self.gamma = GAMMA
        self.count = 0
        self.tau = TAU

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [agent.actor for agent in self.agents]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [agent.target_actor for agent in self.agents]
        return target_actors

    def act(self, obs_all_agents):
        """get actions from all the agents in the MADDPG object"""
        actions = [agent.act(obs, self.epsilon) for agent, obs in zip(self.agents, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents):
        """get target actions from all the agents in the MADDPG object """
        target_actions = [agent.target_act(obs) for agent, obs in zip(self.agents, obs_all_agents)]
        return target_actions

    def get_obs(self, states):
        """Create obs and obs_full from states"""

        # states = states.reshape((2, 3, 8))[:, -1, :-2]
        # states = states * OBSNORM[None, :]
        states = states.reshape((2, 3, 8))[:, :, :-2]
        states = states * OBSNORM[None, None, :]
        # obs1 = Obs(*states[0])
        # obs2 = Obs(*states[1])
        # obs = np.array([[obs1.vx, obs1.vy, obs2.px - obs1.px, obs2.py - obs1.py,
        #                  obs2.vx, obs2.vy, obs1.bx - obs1.px, obs1.by - obs1.py],
        #                 [obs2.vx, obs2.vy, obs1.px - obs2.px, obs1.py - obs2.py,
        #                  obs1.vx, obs1.vy, obs2.bx - obs2.px, obs2.by - obs2.py]])
        # obs = np.array([states[0] - states[1], states[1] - states[0]])
        # flip_array = np.ones((10,))
        # flip_array[0] = -1
        # flip_array[2] = -1
        # obs_full = np.concatenate([states[0, :4], states[1]]) * flip_array
        obs = states.reshape((2, -1))
        obs_full = states.reshape((1, -1))[0]
        # obs = states
        # obs_full = np.concatenate([states[0, :4], states[1]])

        return obs[None, :], obs_full[None, :]

    def learn(self, samples, agent_number, logger=None):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)

        # ---------------------------- update critic ---------------------------- #
        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)

        agent = self.agents[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)

        target_actions = torch.cat(target_actions, dim=1)
        target_critic_input = torch.cat((next_obs_full.t(), target_actions), dim=1).to(device)

        with torch.no_grad():
            q_targets_next = agent.target_critic(target_critic_input)

        q_targets = reward[agent_number].view(-1, 1) + self.gamma * q_targets_next * (1 - done[agent_number].view(-1, 1))

        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)

        q_expected = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q_expected, q_targets.detach())
        # critic_loss = F.mse_loss(q_expected, q_targets)
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        actions_pred = [self.agents[i].actor(ob) if i == agent_number else self.agents[i].actor(ob).detach() for i, ob in enumerate(obs)]

        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        actions_pred = torch.cat(actions_pred, dim=1)
        critic_input2 = torch.cat((obs_full.t(), actions_pred), dim=1)

        # get the policy gradient
        actor_loss = -agent.critic(critic_input2).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
        agent.actor_optimizer.step()


    def reset(self):
        self.count += 1
        self.epsilon = np.exp(-0.0005 * self.count)
        for agent in self.agents:
            agent.noise.reset()

    def update_targets(self):
        """soft update targets"""
        # self.iter += 1
        for agent in self.agents:
            policy_update(agent.actor, agent.target_actor, self.tau)
            policy_update(agent.critic, agent.target_critic, self.tau)

    def save_model(self, filename):
        save_dict_list = []
        for agent in self.agents:
            save_dict = {'actor_params': agent.actor.state_dict(),
                         'actor_optim_params': agent.actor_optimizer.state_dict(),
                         'critic_params': agent.critic.state_dict(),
                         'critic_optim_params': agent.critic_optimizer.state_dict()}
            save_dict_list.append(save_dict)
        torch.save(save_dict_list, filename)


class DDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic):
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # initialize targets same as original networks
        policy_update(self.actor, self.target_actor, 1)
        policy_update(self.critic, self.target_critic, 1)

        self.noise = OUNoise(out_actor)

    def act(self, obs, epsilon):
        obs = obs.to(device)
        action = self.actor(obs)
        if random.random() < epsilon:
            action += self.noise.sample()
        return action

    def target_act(self, obs):
        obs = obs.to(device)
        action = self.target_actor(obs)
        return action


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state).float()


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
