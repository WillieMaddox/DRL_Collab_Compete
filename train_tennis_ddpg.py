import os
import time
from collections import deque
import numpy as np
from tennis_ddpg import UnityTennisEnv, Agent
from tennis_ddpg import ddpg_distance_metric
from tennis_ddpg import PSNoise

WEIGHT_DECAY = 0    # L2 weight decay
NOISE_DECAY = 0.99995   #
PRELOAD_STEPS = int(1e4)  # initialize the replay buffer with this many transitions.
BUFFER_SIZE = int(2e5)    # replay buffer size
BATCH_SIZE = 256          # minibatch size
GAMMA = 0.99              # discount factor
TAU = 0.02                # for soft update of target parameters
LR_ACTOR = 2e-4           # Learning rate of the actor
LR_CRITIC = 2e-3          # Learning rate of the critic
UPDATE_EVERY = 1          # Update the network after this many steps.
LEARN_EVERY = 1           # Train local network ever n-steps
NUM_EPISODES = 4000


def train(env, agent, preload_steps=PRELOAD_STEPS, n_episodes=NUM_EPISODES, t_max=2000, print_interval=100):
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
    early_stop = 0.5

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
        agent.buffer.push(transition)
        obs = obs_next
        if dones.any():
            obs = env.reset()

    param_noise = PSNoise()

    for i_episode in range(1, n_episodes + 1):
        episode_rewards = np.zeros((env.num_agents,))  # initialize the score (for each agent)
        obs = env.reset()  # reset the environment
        agent.reset()
        agent.perturb_actor_parameters(param_noise)

        t_step = 0
        while True:

            actions = agent.act(obs.reshape(-1))  # based on the current state get an action.
            obs_next, rewards, dones = env.step(actions.reshape(2, -1))  # send all actions to the environment
            transition = (obs.reshape((1, -1)), actions.reshape((1, -1)), np.max(rewards, keepdims=True).reshape((1, -1)), obs_next.reshape((1, -1)), np.max(dones, keepdims=True).reshape((1, -1)))
            agent.step(transition)
            obs = obs_next
            episode_rewards += rewards  # update the score (for each agent)
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

        perturbed_states, perturbed_actions, _, _, _ = agent.buffer.tail(t_step)
        unperturbed_actions = agent.act(np.array(perturbed_states), False, False)
        ddpg_dist = ddpg_distance_metric(np.array(perturbed_actions), unperturbed_actions)
        param_noise.adapt(ddpg_dist)



        summary = f'\rEpisode {i_episode:>4}  Buffer Size: {len(agent.buffer):>6}  Noise: {agent.noise_scale:.2f}  t_step: {t_step:4}  Episode Score (Avg): {episode_reward:.2f} ({mean:.3f})'
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
        'noise_decay': NOISE_DECAY,
    }

    agent = Agent(state_size, action_size, **agent_config)
    scores, scores_average = train(env, agent)
    print(time.time() - t0, 'seconds')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.plot(np.arange(1, len(scores_average) + 1), scores_average)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    env.close()
