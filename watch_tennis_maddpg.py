import time
import argparse
import numpy as np
from train_tennis_maddpg import UnityTennisEnv, Agent


def watch(env, agent1, agent2, episodes):
    scores = []
    for ep_i in range(episodes):
        score = np.zeros((2, 2))
        obs = env.reset(train_mode=False)
        while True:
            actions1 = agent1.act(obs[0], train_mode=False, perturb_mode=False)[0]
            actions2 = agent2.act(obs[1], train_mode=False, perturb_mode=False)[0]
            obs_next, rewards, dones = env.step([actions1, actions2])
            score += rewards
            obs = obs_next
            if dones.any():
                break
        scores.append(np.max(score))
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent1', '-a1', default='model_dir/tennis/maddpg-1548748683-best.pt')
    parser.add_argument('--agent2', '-a2', default='model_dir/tennis/maddpg-1548748683-best.pt')
    parser.add_argument('--n_episodes', '-n', default=10)
    args = parser.parse_args()

    # create environment
    env = UnityTennisEnv(file_name='Tennis_Linux/Tennis.x86_64', no_graphics=False)

    # restore agent checkpoint
    agent1 = Agent(env.state_size, env.action_size, 0, restore=args.agent1)
    agent2 = Agent(env.state_size, env.action_size, 1, restore=args.agent2)

    t0 = time.time()
    scores = watch(env, agent1, agent2, args.n_episodes)
    print(f'Average score over {args.n_episodes} episodes: {np.mean(scores):.2f}, {time.time()-t0:.2f} seconds')

    env.close()
