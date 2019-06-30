import argparse
import os
import random
import sys

import numpy as np
import torch

from unityagents import UnityEnvironment

import DDPG
import mDDPG
import OurDDPG
import TD3
import utils


class UnityWrapper:
    def __init__(self, env, train_mode=True):
        self.env = env
        self.brain_name = env.brain_names[0]
        self.train_mode = train_mode

        brain = env.brains[self.brain_name]
        action_size = brain.vector_action_space_size

        self.action_space = action_size
        self.observation_space = brain.vector_observation_space_size
        self.num_agents = len(self.reset())

        self.action_space_low = -1
        self.action_space_high = 1
        self._max_episode_steps = 1000
        self.num_agents = 1

    def reset(self, train_mode=None):
        if train_mode is None:
            train_mode = self.train_mode
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        return env_info.vector_observations[0]

    def step(self, actions):
        env_info = self.env.step(vector_action=actions)[self.brain_name]
        rewards = env_info.rewards  # get reward (for each agent)
        dones = [1 if x else 0 for x in env_info.local_done]  # see if episode finished
        next_states = env_info.vector_observations  # get next state (for each agent)
        return next_states[0], rewards[0], dones[0], None

    def sample(self):
        return [random.uniform(self.action_space_low, self.action_space_high) for _ in range(self.action_space)]

    def close(self):
        return self.env.close()

    def seed(self, seed):
        pass


def executable():
    if sys.platform == "linux":
        return "Reacher_Linux_NoVis/Reacher.x86_64"
    if sys.platform == "darwin":
        return "Reacher.app"


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, eval_episodes=10):
    rewards = []
    for _ in range(eval_episodes):
        episode_reward = 0
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)

    rewards = np.array(rewards)
    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, rewards.mean()))
    print("---------------------------------------")
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="Reacher-v2")
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e3, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    for dirname in ("./results", "./rewards"):
        os.makedirs(dirname, exist_ok=True)
    if args.save_models:
        os.makedirs("./pytorch_models", exist_ok=True)

    unity = UnityEnvironment(file_name=executable(), no_graphics=True)
    env = UnityWrapper(unity, train_mode=True)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space
    action_dim = env.action_space
    max_action = 1

    # Initialize policy
    if args.policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG":
        policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "mDDPG":
        policy = mDDPG.DDPG(state_dim, action_dim, max_action)

    replay_buffer = utils.ReplayBuffer()

    # Evaluate untrained policy
    evaluations = [evaluate_policy(env, policy).mean()]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = True
    rewards = []

    while total_timesteps < args.max_timesteps:
        if done:
            if total_timesteps != 0:
                print(
                    ("Total T: %d Episode Num: %d Episode T: %d Reward: %f")
                    % (total_timesteps, episode_num, episode_timesteps, episode_reward)
                )
                rewards.append(episode_reward)
                if args.policy_name == "TD3":
                    policy.train(
                        replay_buffer,
                        episode_timesteps,
                        args.batch_size,
                        args.discount,
                        args.tau,
                        args.policy_noise,
                        args.noise_clip,
                        args.policy_freq,
                    )
                else:
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(env, policy).mean())

                if args.save_models:
                    policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space)).clip(
                    env.action_space_low, env.action_space_high
                )

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(env, policy, 100).mean())
    if args.save_models:
        policy.save("%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)
    np.save("./rewards/%s" % (file_name), rewards)


if __name__ == "__main__":
    main()
