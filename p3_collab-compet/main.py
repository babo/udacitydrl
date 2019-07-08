import argparse
import os
import sys

import numpy as np
import torch

from unityagents import UnityEnvironment

import mDDPG
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
        self.observation_space = brain.num_stacked_vector_observations * brain.vector_observation_space_size
        self.num_agents = self.reset().shape[0]

        self.action_space_low = -1
        self.action_space_high = 1
        self._max_episode_steps = 10000

    def reset(self, train_mode=None):
        if train_mode is None:
            train_mode = self.train_mode
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        env_info = self.env.step(vector_action=actions)[self.brain_name]
        rewards = env_info.rewards  # get reward (for each agent)
        dones = [1 if x else 0 for x in env_info.local_done]  # see if episode finished
        next_states = env_info.vector_observations  # get next state (for each agent)
        return next_states, rewards, dones

    def sample(self):
        return np.random.uniform(self.action_space_low, self.action_space_high, (self.num_agents, self.action_space))

    def close(self):
        return self.env.close()


def executable():
    if sys.platform == "linux":
        return "Tennis_Linux/Tennis.x86_64"
    if sys.platform == "darwin":
        return "Tennis.app"


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, eval_episodes=10):
    all_rewards = []
    for _ in range(eval_episodes):
        episode_reward = np.zeros(env.num_agents)
        observations = env.reset()
        done = False
        while not done:
            actions = [policy.select_action(np.array(obs)) for obs in observations]
            observations, rewards, dones = env.step(actions)
            episode_reward += rewards
            done = np.any(dones)
        all_rewards.append(episode_reward.max())

    rewards = np.array(all_rewards).mean()
    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, rewards))
    print("---------------------------------------")
    return rewards


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="Tennis")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    for dirname in ("./results", "./rewards"):
        os.makedirs(dirname, exist_ok=True)
    if args.save_models:
        os.makedirs("./pytorch_models", exist_ok=True)

    unity = UnityEnvironment(file_name=executable(), no_graphics=True)
    try:
        env = UnityWrapper(unity, train_mode=True)

        # Set seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = env.observation_space
        action_dim = env.action_space
        max_action = 1

        # Initialize policy
        if args.policy_name == "TD3":
            pd = TD3.TD3
        elif args.policy_name == "mDDPG":
            pd = mDDPG.DDPG
        else:
            print("Unknown policy: {}".format(args.policy_name))
            return
        policy = pd(state_dim, action_dim, max_action)

        replay_buffer = utils.ReplayBuffer()

        # Evaluate untrained policy
        evaluations = [evaluate_policy(env, policy)]

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        episode_reward = np.zeros(env.num_agents)
        episode_timesteps = 0
        done = True
        all_rewards = []

        while total_timesteps < args.max_timesteps:
            if done:
                if total_timesteps != 0:
                    all_rewards.append(episode_reward.max())
                    print(
                        ("Total T: %d Episode Num: %d Episode T: %d Reward: %f")
                        % (total_timesteps, episode_num, episode_timesteps, episode_reward.max())
                    )
                    if total_timesteps > 1000:
                        ts = 10  # min(1000, total_timesteps) // batch_size
                    elif total_timesteps > args.batch_size:
                        ts = 1
                    else:
                        ts = 0

                    if args.policy_name == "TD3":
                        policy.train(
                            replay_buffer,
                            ts,
                            args.batch_size,
                            args.discount,
                            args.tau,
                            args.policy_noise,
                            args.noise_clip,
                            args.policy_freq,
                        )
                    else:
                        policy.train(replay_buffer, ts, args.batch_size, args.discount, args.tau)

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    evaluations.append(evaluate_policy(env, policy))

                    if args.save_models:
                        policy.save(file_name, directory="./pytorch_models")
                    np.save("./results/%s" % (file_name), evaluations)
                    np.save("./rewards/%s" % (file_name), all_rewards)

                # Reset environment
                observations = env.reset()
                done = False
                episode_reward = np.zeros(env.num_agents)
                episode_timesteps = 0
                episode_num += 1

            # Select action randomly or according to policy
            if total_timesteps < args.start_timesteps:
                actions = env.sample()
            else:
                actions = [policy.select_action(np.array(obs)) for obs in observations]
                if args.expl_noise != 0:
                    actions = (actions + np.random.normal(0, args.expl_noise, size=(env.num_agents, env.action_space))).clip(
                        env.action_space_low, env.action_space_high
                    )

            # Perform action
            new_obs, rewards, dones = env.step(actions)
            done = np.any(dones)
            if episode_timesteps + 1 == env._max_episode_steps:
                dones = np.zeros(env.num_agents)
            episode_reward += rewards

            # Store data in replay buffer
            for x, y, a, r, d in zip(observations, new_obs, actions, rewards, dones):
                replay_buffer.add((x, y, a, r, d))

            observations = new_obs

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        # Final evaluation
        evaluations.append(evaluate_policy(env, policy, 100))
        if args.save_models:
            policy.save(file_name, directory="./pytorch_models")
        np.save("./results/%s" % (file_name), evaluations)
        np.save("./rewards/%s" % (file_name), all_rewards)
    finally:
        unity.close()


if __name__ == "__main__":
    main()
