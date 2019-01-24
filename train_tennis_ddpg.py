import os
import time
from collections import deque
import numpy as np
from tennis_ddpg import UnityEnv, Agent

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


def train(env, agent, preload_steps=0, n_episodes=2000, t_max=1000, print_interval=100):
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
    best = 0
    early_stop = 0.61

    # log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir"
    os.makedirs(model_dir, exist_ok=True)

    print('BUFFER_SIZE:', BUFFER_SIZE)

    # fill replay buffer with rnd actions
    obs = env.reset()
    for _ in range(preload_steps):
        actions = np.random.randn(2, 2)  # select an action (for each agent)
        obs_next, rewards, dones = env.step(actions)
        transition = (obs.reshape((1, -1)), actions.reshape((1, -1)), np.max(rewards, keepdims=True).reshape((1, -1)), obs_next.reshape((1, -1)), np.max(dones, keepdims=True).reshape((1, -1)))
        # transition = (obs.reshape(-1), actions.reshape(-1), np.max(rewards, keepdims=True).reshape(-1), obs_next.reshape(-1), np.max(dones, keepdims=True).reshape(-1))
        agent.buffer.push(transition)
        obs = obs_next
        if dones.any():
            obs = env.reset()

    for i_episode in range(1, n_episodes + 1):
        episode_rewards = np.zeros((env.num_agents,))  # initialize the score (for each agent)
        obs = env.reset()  # reset the environment
        agent.reset()
        t_step = 0

        while True:

            actions = agent.act(obs.reshape(-1))  # based on the current state get an action.
            obs_next, rewards, dones = env.step(actions.reshape(2, -1))  # send all actions to the environment
            # preloaded = t_step >= 2
            # push_info = random.random() < 1.0
            # on_reward = np.sum(np.abs(rewards)) > 1e-8
            # if preloaded and (push_info or on_reward):
            #     transition = (obs, actions, rewards, obs_next, dones)
            #     buffer.push(transition)
            transition = (obs.reshape((1, -1)), actions.reshape((1, -1)), np.max(rewards, keepdims=True).reshape((1, -1)), obs_next.reshape((1, -1)), np.max(dones, keepdims=True).reshape((1, -1)))
            # transition = (obs.reshape(-1), actions.reshape(-1), np.max(rewards, keepdims=True).reshape(-1), obs_next.reshape(-1), np.max(dones, keepdims=True).reshape(-1))
            agent.step(transition)
            obs = obs_next
            episode_rewards += rewards  # update the score (for each agent)

            # if np.any(env.max_reached):
            #     print(t_step)
            #     print(env.max_reached)
            #     print(dones)

            if dones.any():  # exit loop if episode finished
                break

            t_step += 1  # increment the number of steps seen this episode.
            if t_step >= t_max:  # exit loop if episode finished
                break

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

        summary = f'\rEpisode {i_episode:>4}  Buffer Size: {len(agent.buffer):>6}  Noise: {agent.noise_scale:.2f}  Eps: {agent.epsilon:.4f}  t_step: {t_step:4}  Episode Score (Avg): {episode_reward:.2f} ({mean:.3f})'

        if mean >= 0.5 and mean > best:
            summary += " (saved)"
            best = mean
            agent.save_model(os.path.join(model_dir, f'tennis-episode-{i_episode}.pt'))

        if i_episode % print_interval == 0:
            print(summary)
        else:
            print(summary, end="")

        if best > early_stop:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {mean:.2f}')
            break

    return scores, scores_average




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = UnityEnv(file_name='Tennis_Linux/Tennis.x86_64', no_graphics=True)
    state_size = env.num_agents * env.state_size
    action_size = env.num_agents * env.action_size

    t0 = time.time()
    agent = Agent(state_size, action_size)
    scores, scores_average = train(env, agent, int(1e4), n_episodes=10000, t_max=2000)
    print(time.time() - t0, 'seconds')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.plot(np.arange(1, len(scores_average) + 1), scores_average)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    env.close()
