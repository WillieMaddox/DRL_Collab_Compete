import os
import time
from collections import deque
import numpy as np
from tennis_ddpg import UnityTennisEnv, Agent

PRELOAD_STEPS = int(1e4)  # initialize the replay buffer with this many transitions.
BUFFER_SIZE = int(2e5)    # replay buffer size
BATCH_SIZE = 256          # minibatch size
GAMMA = 0.99              # discount factor
TAU = 0.02                # for soft update of target parameters
LR_ACTOR = 2e-4           # Learning rate of the actor
LR_CRITIC = 2e-3          # Learning rate of the critic
UPDATE_EVERY = 1          # Update the network after this many steps.
LEARN_EVERY = 1           # Train local network ever n-steps
RANDOM_SEED = 0
NUM_EPISODES = 4000

USE_ASN = True  # Use Action Space Noise
ASN_KWARGS = {
    'mu': 0.0,
    'theta': 0.15,
    'sigma': 0.20,
    'scale_start': 1.0,
    'scale_end': 0.01,
    'decay': 0.99995
}

USE_PSN = True  # Use Parameter Space Noise
PSN_KWARGS = {
    'initial_stddev': 0.1,
    'desired_action_stddev': 0.1,
    'adoption_coefficient': 1.01
}

USE_PER = True  # Use Prioritized Experience Replay
PER_PRIORITY_START = 1.0
PER_PRIORITY_END = 0.3
PER_PRIORITY_DECAY = 0.9999


def train(env, agent, preload_steps=PRELOAD_STEPS, n_episodes=NUM_EPISODES, print_interval=100):
    """Train using Deep Deterministic Policy Gradients.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        t_max (int): maximum number of timesteps per episode
        print_every (int): print after this many episodes. Also used to define length of the deque buffer.
    """
    pri = PER_PRIORITY_START
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    scores_average = []
    best = 0
    early_stop = 0.5

    # log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir/tennis"
    os.makedirs(model_dir, exist_ok=True)

    print('BUFFER_SIZE:', BUFFER_SIZE)

    # fill replay buffer with rnd actions
    obs = env.reset()
    for _ in range(preload_steps):
        actions = np.random.randn(4)  # select an action (for each agent)
        obs_next, rewards, dones = env.step(actions)
        transition = (obs, actions, np.sum(rewards, keepdims=True), obs_next, np.max(dones, keepdims=True))
        agent.buffer.push(transition)
        obs = obs_next
        if dones.any():
            obs = env.reset()

    for i_episode in range(1, n_episodes + 1):
        episode_rewards = np.zeros((1, env.num_agents))  # initialize the score (for each agent)
        obs = env.reset()  # reset the environment
        agent.reset()

        t_step = 0
        pri = max(pri * PER_PRIORITY_DECAY, PER_PRIORITY_END)
        while True:
            actions = agent.act(obs)  # based on the current state get an action.
            obs_next, rewards, dones = env.step(actions)  # send all actions to the environment
            transition = (obs, actions, np.sum(rewards, keepdims=True), obs_next, np.max(dones, keepdims=True))

            agent.step(transition, pri)
            obs = obs_next
            episode_rewards += rewards  # update the score (for each agent)
            if dones.any():  # exit loop if episode finished
                break
            t_step += 1  # increment the number of steps seen this episode.

        episode_reward = np.max(episode_rewards)
        scores_window.append(episode_reward)  # save most recent score
        scores.append(episode_reward)
        mean = np.mean(scores_window)
        scores_average.append(mean)

        agent.postprocess(t_step)

        summary = f'\rEpisode {i_episode:>4}  Buffer Size: {len(agent.buffer):>6}  Noise: {agent.action_noise.scale:.2f}  t_step: {t_step:4}  Score (Avg): {episode_reward:.2f} ({mean:.3f})'

        if mean >= 0.5 and mean > best:
            summary += " (saved)"
            best = mean
            agent.save_model(model_dir, env.session_name, i_episode, best)

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

    env = UnityTennisEnv(file_name='Tennis_Linux/Tennis.x86_64', no_graphics=True)
    state_size = env.num_agents * env.state_size
    action_size = env.num_agents * env.action_size

    t0 = time.time()
    agent_config = {
        'buffer_size': BUFFER_SIZE,
        'batch_size': BATCH_SIZE,
        'learn_every': LEARN_EVERY,
        'update_every': UPDATE_EVERY,
        'gamma': GAMMA,
        'tau': TAU,
        'lr_actor': LR_ACTOR,
        'lr_critic': LR_CRITIC,
        'random_seed': RANDOM_SEED,
        'use_asn': USE_ASN,
        'asn_kwargs': ASN_KWARGS,
        'use_psn': USE_PSN,
        'psn_kwargs': PSN_KWARGS,
        'use_per': USE_PER
    }
    agent = Agent(state_size, action_size, **agent_config)

    print('session_name', env.session_name)
    scores, scores_average = train(env, agent)
    print(time.time() - t0, 'seconds')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.plot(np.arange(1, len(scores_average) + 1), scores_average)
    ax.axhline(y=0.5, xmin=0.0, xmax=1.0, linestyle='--')
    plt.title(f'Final Buffer Length {len(agent.buffer)}')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    filename = f'model_dir/tennis/ddpg_{env.session_name}'
    filename += f'-PER' if USE_PER else f'-ER'
    filename += f'_{PRELOAD_STEPS:d}'
    if USE_PSN:
        filename += f'-PSN'
    plt.savefig(filename)
    plt.show()

    env.close()
