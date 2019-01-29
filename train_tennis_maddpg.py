import os
import numpy as np
from collections import namedtuple, deque

import torch

from tennis_maddpg import MADDPG, ReplayBuffer
from utils import transpose_to_tensor

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


def train(multi_agent, n_episodes=2000, t_max=1000, print_interval=200, update_interval=2):
    """Train using Deep Deterministic Policy Gradients.

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
    early_stop = 0.7
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
                break

            t_step += 1  # increment the number of steps seen this episode.
            if t_step >= t_max:  # exit loop if episode finished
                break

        scores.append(np.max(episode_rewards))
        scores_window.append(np.max(episode_rewards))  # save most recent score
        mean = np.mean(scores_window)
        scores_average.append(mean)

        if len(buffer) > BATCH_SIZE and update_info:
            for a_i in range(2):
                samples = buffer.sample()
                multi_agent.learn(samples, a_i)
            multi_agent.update_targets()  # soft update the target network towards the actual networks

        summary = '\rEpisode {:>4}  Eps: {:.4f}  t_steps: {:4}  Current Score: {:.2f}  Average Score: {:.3f}'.format(i_episode, multi_agent.epsilon, t_step, scores[-1], mean)

        if mean >= 0.5 and mean > best:
            summary += " (saved)"
            best = mean
            multi_agent.save_model(os.path.join(model_dir, 'tennis-episode-{}.pt'.format(i_episode)))

        if print_info:
            print(summary)
        else:
            print(summary, end="")

        if best > early_stop:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, mean))
            break

    return scores, scores_average


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from unityagents import UnityEnvironment

    # env = UnityEnvironment(file_name="./Tennis_Linux_NoVis/Tennis.x86_64")
    env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64', no_graphics=True)
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
    multi_agent = MADDPG(state_size, action_size, num_agents)
    preload_replay_buffer(env, multi_agent, int(1e4))
    scores, scores_average = train(multi_agent, n_episodes=6000, t_max=1000)
    print(time.time() - t0, 'seconds')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.plot(np.arange(1, len(scores_average) + 1), scores_average)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    env.close()
