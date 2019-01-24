import os
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

        # al = actor_loss.cpu().detach().item()
        # cl = critic_loss.cpu().detach().item()
        # logger.add_scalars('agent%i/losses' % agent_number,
        #                    {'critic loss': cl,
        #                     'actor_loss': al},
        #                    self.iter)

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


class Agent0:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents):
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

        # Hard copy weights from local to target networks
        policy_update(self.actor_local, self.actor_target, 1.0)
        policy_update(self.critic_local, self.critic_target, 1.0)

        # Noise process
        # self.noise = OUNoise0(num_agents, action_size)
        self.noise = OUNoise0(None, action_size)
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


class OUNoise0:
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


# fill replay buffer with rnd actions
def preload_replay_buffer(env, agent, steps):
    env_info = env.reset(train_mode=True)[brain_name]
    obs = agent.get_obs(env_info.vector_observations)

    for _ in range(steps):
        action = np.random.randn(2, 2)  # select an action (for each agent)
        action = np.clip(action, -1, 1)  # all actions between -1 and 1
        env_info = env.step(action)[brain_name]
        obs_next = agent.get_obs(env_info.vector_observations)

        reward = np.array(env_info.rewards)
        done = np.array(env_info.local_done).astype(np.float)

        transition = (obs.reshape((1, -1)), action.reshape((1, -1)), np.max(reward, keepdims=True).reshape((1, -1)), obs_next.reshape((1, -1)), np.max(done, keepdims=True).reshape((1, -1)))
        # transition = (obs.reshape(-1), action.reshape(-1), np.max(reward, keepdims=True).reshape(-1), obs_next.reshape(-1), np.max(done, keepdims=True).reshape(-1))

        agent.buffer.push(transition)

        obs = obs_next
        if done.any():
            env_info = env.reset(train_mode=True)[brain_name]
            obs = agent.get_obs(env_info.vector_observations)


def ddpg(agent, n_episodes=2000, t_max=1000, print_interval=100):
    """Deep Deterministic Policy Gradients.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        t_max (int): maximum number of timesteps per episode
        print_every (int): print after this many episodes. Also used to define length of the deque buffer.
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    scores_average = []
    parallel_envs = 1
    best = 0
    early_stop = 0.61

    # log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir"
    os.makedirs(model_dir, exist_ok=True)

    # buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
    print('BUFFER_SIZE:', BUFFER_SIZE)

    for i_episode in range(1, n_episodes + 1):
        episode_rewards = np.zeros((num_agents,))  # initialize the score (for each agent)
        agent.reset()
        t_step = 0

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        obs = agent.get_obs(env_info.vector_observations)  # get the current state (for each agent)

        print_info = i_episode % print_interval < parallel_envs
        # update_info = i_episode % update_interval < parallel_envs

        while True:

            actions = agent.act(obs.reshape(-1))  # based on the current state get an action.

            env_info = env.step(actions.reshape(2, -1))[brain_name]  # send all actions to the environment
            obs_next = agent.get_obs(env_info.vector_observations)  # get obs from next state

            rewards = np.array(env_info.rewards)  # get reward (for each agent)
            dones = np.array(env_info.local_done).astype(np.float)  # see if episodes finished

            # preloaded = t_step >= 2
            # push_info = random.random() < 1.0
            # on_reward = np.sum(np.abs(rewards)) > 1e-8
            # if preloaded and (push_info or on_reward):
            #     transition = (obs, actions, rewards, obs_next, dones)
            #     buffer.push(transition)

            transition = (obs.reshape((1, -1)), actions.reshape((1, -1)), np.max(rewards, keepdims=True).reshape((1, -1)), obs_next.reshape((1, -1)), np.max(dones, keepdims=True).reshape((1, -1)))
            # transition = (obs.reshape(-1), actions.reshape(-1), np.max(rewards, keepdims=True).reshape(-1), obs_next.reshape(-1), np.max(dones, keepdims=True).reshape(-1))
            # transition = (obs, actions, rewards, obs_next, dones)
            agent.step(transition)
            # buffer.push(transition)

            obs = obs_next
            episode_rewards += rewards  # update the score (for each agent)

            if np.any(dones):  # exit loop if episode finished
                break

            t_step += 1  # increment the number of steps seen this episode.
            if t_step >= t_max:  # exit loop if episode finished
                break

            if np.any(env_info.max_reached):
                print(t_step)
                print(env_info.max_reached)
                raise ValueError

        episode_reward = np.max(episode_rewards)
        scores_window.append(episode_reward)  # save most recent score
        scores.append(episode_reward)
        mean = np.mean(scores_window)
        scores_average.append(mean)

        # If enough samples are available in memory, get random subset and learn
        # if update_info and len(buffer) > BATCH_SIZE * NUM_BATCHES:
        #     for _ in range(NUM_BATCHES):
        #         samples = buffer.sample()
        #         agent.learn(samples)
        #         agent.update_targets()  # soft update the target network towards the actual networks

        summary = f'\rEpisode {i_episode:>4}  Buffer Size: {len(agent.buffer):>6}  Noise: {agent.noise_scale:.2f}  Eps: {agent.epsilon:.4f}  t_step: {t_step:4}  Episode Score: {episode_reward:.2f}  Average Score: {mean:.3f}'

        if mean >= 0.5 and mean > best:
            summary += " (saved)"
            best = mean
            agent.save_model(os.path.join(model_dir, f'tennis-episode-{i_episode}.pt'))

        if print_info:
            print(summary)
        else:
            print(summary, end="")

        if best > early_stop:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {mean:.2f}')
            break

    return scores, scores_average


def maddpg(multi_agent, n_episodes=2000, t_max=1000, save_interval=1000, print_interval=200, update_interval=2):
    """Deep Deterministic Policy Gradients.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        t_max (int): maximum number of timesteps per episode
        print_every (int): print after this many episodes. Also used to define length of the deque buffer.
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    parallel_envs = 1

    # log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir"
    os.makedirs(model_dir, exist_ok=True)

    buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        obs, obs_full = multi_agent.get_obs(states)
        multi_agent.reset()
        episode_rewards = np.zeros((1, num_agents))  # initialize the score (for each agent)
        t_step = 0

        save_info = i_episode % save_interval < parallel_envs
        print_info = i_episode % print_interval < parallel_envs
        update_info = i_episode % update_interval < parallel_envs

        while True:

            actions = multi_agent.act(transpose_to_tensor(obs))  # based on the current state get an action.
            actions_array = torch.stack(actions).detach().numpy()
            actions_clipped = np.clip(actions_array, -1, 1)
            actions_for_env = np.rollaxis(actions_clipped, 1)

            env_info = env.step(actions_for_env[0])[brain_name]  # send all actions to the environment
            states_next = env_info.vector_observations  # get next state (for each agent)
            obs_next, obs_full_next = multi_agent.get_obs(states_next)
            rewards = np.array([env_info.rewards]).reshape((1, -1))  # get reward (for each agent)
            dones = np.array([env_info.local_done]).reshape((1, -1))  # see if episodes finished

            if t_step >= 2:
                transition = (obs, obs_full, actions_for_env, rewards, obs_next, obs_full_next, dones)
                buffer.push(transition)

            obs, obs_full = obs_next, obs_full_next  # roll over states to next time step
            episode_rewards += rewards  # update the score (for each agent)

            if np.any(dones):  # exit loop if episode finished
                # print('here')
                break

            t_step += 1  # increment the number of steps seen this episode.
            if t_step >= t_max:  # exit loop if episode finished
                # episode_rewards = episode_rewards * 1000.0 / t_step
                break

            if np.any(env_info.max_reached):
                print(t_step)
                print(env_info.max_reached)
                raise ValueError

        # if np.any(episode_rewards >= 0.15):  # exit loop if episode finished
        #     print('here')

        scores.append(np.max(episode_rewards))
        scores_window.append(np.max(episode_rewards))  # save most recent score

        if len(buffer) > BATCH_SIZE and update_info:
            for a_i in range(2):
                samples = buffer.sample()
                multi_agent.learn(samples, a_i)
            multi_agent.update_targets()  # soft update the target network towards the actual networks

        print('\rEpisode {:>4}\tCurrent Score: {:.2f}\tAverage Score: {:.2f}\tt_step: {:4}\tEps: {:.4f}'.format(i_episode, scores[-1], np.mean(scores_window), t_step, multi_agent.epsilon), end="")
        if print_info:
            print('\rEpisode {:>4}\tCurrent Score: {:.2f}\tAverage Score: {:.2f}\tt_step: {:4}\tEps: {:.4f}'.format(i_episode, scores[-1], np.mean(scores_window), t_step, multi_agent.epsilon))

        if save_info:
            multi_agent.save_model(os.path.join(model_dir, 'episode-{}.pt'.format(i_episode)))

        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            multi_agent.save_model(os.path.join(model_dir, 'episode-{}.pt'.format(i_episode)))
            break

    return scores


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from unityagents import UnityEnvironment

    env = UnityEnvironment(file_name="./Tennis_Linux_NoVis/Tennis.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = 2 * brain.vector_action_space_size
    state_size = 2 * (env_info.vector_observations.shape[1] - 6)
    t0 = time.time()
    agent = Agent0(state_size, action_size, num_agents)
    preload_replay_buffer(env, agent, int(1e4))
    scores, scores_average = ddpg(agent, n_episodes=10000, t_max=2000)
    # multi_agent = MADDPG(state_size, action_size, num_agents)
    # scores = maddpg(multi_agent, n_episodes=6000, t_max=1000, update_interval=6)
    print(time.time() - t0, 'seconds')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.plot(np.arange(1, len(scores_average) + 1), scores_average)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    env.close()
