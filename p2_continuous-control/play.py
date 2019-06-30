#!/usr/bin/env python


import numpy as np

from unityagents import UnityEnvironment

import TD3

from main import UnityWrapper


def executable():
    return "Reacher_Linux_20/Reacher.x86_64"


def play():
    unity = UnityEnvironment(file_name=executable(), no_graphics=False)
    env = UnityWrapper(unity, train_mode=False)

    policy = TD3.TD3(env.observation_space, env.action_space, 1)
    policy.load("TD3_Reacher-v2_3", directory="pytorch_models")

    for _ in range(5):
        obs = env.reset()
        done = False
        episode_score = np.zeros(20, dtype=np.float64)
        while not done:
            action = [policy.select_action(np.array(observation)) for observation in obs]
            obs, reward, done, _ = env.step(action)
            episode_score += reward
            done = np.any(done)
        print("Scored: {:.2f}".format(episode_score.mean()))

    unity.close()


play()


if __name__ == "__main__":
    play()
