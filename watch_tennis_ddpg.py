# pylint: skip-file
import time
import argparse
import numpy as np
from train_tennis_ddpg import UnityTennisEnv, Agent


def watch(env, agent, episodes):
    scores = []
    for ep_i in range(episodes):
        score = np.zeros((2, 2))
        obs = env.reset(train_mode=False)
        while True:
            action = agent.act(obs, train_mode=False, perturb_mode=False)
            obs_next, rewards, dones = env.step(action)
            score += rewards
            obs = obs_next
            if dones.any():
                break
        scores.append(np.max(score))
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', '-a', default='model_dir/tennis/ddpg_1548835777-best.pt')
    parser.add_argument('--n_episodes', '-n', default=10)
    args = parser.parse_args()

    # create environment
    env = UnityTennisEnv(file_name='Tennis_Linux/Tennis.x86_64', no_graphics=False)
    state_size = env.num_agents * env.state_size
    action_size = env.num_agents * env.action_size

    # restore agent checkpoint
    agent = Agent(state_size, action_size, restore=args.agent)

    t0 = time.time()
    scores = watch(env, agent, args.n_episodes)

    print(f'Average score over {args.n_episodes} episodes: {np.mean(scores):.2f}, {time.time()-t0:.2f} seconds')

    env.close()
