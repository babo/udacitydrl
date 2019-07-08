#!/usr/bin/env python

import numpy as np

from unityagents import UnityEnvironment

import TD3
import mDDPG

from main import executable
from main import UnityWrapper


def play(env):
    p_td3 = TD3.TD3(env.observation_space, env.action_space, 1)
    p_td3.load("TD3_Tennis_12", directory="pytorch_models")
    p_ddpg = mDDPG.DDPG(env.observation_space, env.action_space, 1)
    p_ddpg.load("mDDPG_Tennis_12", directory="pytorch_models")
    policies = [p_ddpg, p_td3]

    scores = []
    for _ in range(100):
        obs = env.reset(train_mode=True)
        done = False
        episode_score = np.zeros(env.num_agents, dtype=np.float64)
        while not done:
            action = [policy.select_action(np.array(observation)) for policy, observation in zip(policies, obs)]
            obs, reward, done = env.step(action)
            episode_score += reward
            done = np.any(done)
        print("Scored: {:.2f} {:.2f}".format(episode_score[0], episode_score[1]))
        scores.append(episode_score.max())
    scores = np.array(scores)
    print("Mean score {:0.2f} over {}".format(scores.mean(), len(scores)))


def main():
    unity = UnityEnvironment(file_name=executable(), no_graphics=False)
    try:
        env = UnityWrapper(unity, train_mode=True)
        play(env)
    finally:
        unity.close()


if __name__ == "__main__":
    main()
