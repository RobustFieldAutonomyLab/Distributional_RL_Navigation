import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    ##### modification #####
    configs: dict = {},
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    assert n_envs == 1, "only support single evaluation environment"
    # episode_rewards = []
    # episode_lengths = []

    ##### modification #####
    reward_data = []
    action_data = []
    success_data = []
    time_data = []
    energy_data = []

    for k, config in enumerate(configs.values()):

        episode_counts = np.zeros(n_envs, dtype="int")
        # Divides episodes among different sub environments in the vector as evenly as possible
        episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
        
        observations = env.reset()
        if config is not None:
            observations = env.reset_with_eval_config(config)
        states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        ##### modification #####
        a = []
        cumulative_reward = 0.0
        length = 0
        energy = 0.0
        done = False

        print(f"Evaluating episode {k}")
        while (episode_counts < episode_count_targets).any():
            actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
            observations, rewards, dones, infos = env.step(actions)
            # current_rewards += rewards
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:

                    # unpack values so that the callback can access the local variables
                    reward = rewards[i]
                    done = dones[i]
                    info = infos[i]
                    episode_starts[i] = done

                    cumulative_reward += env.discount ** length * reward
                    length += 1
                    energy += env.compute_action_energy_cost(int(actions[i]))
                    a.append(int(actions[i]))

                    if callback is not None:
                        callback(locals(), globals())
                    
                    ##### modification #####
                    if done or length >= 1000:
                        if not done and config is None:
                            env.update_episode_data()

                        success = True if info["state"] == "reach goal" else False
                        time = env.T * length

                        action_data.append(a)
                        reward_data.append(cumulative_reward)
                        success_data.append(success)
                        time_data.append(time)
                        energy_data.append(energy)
                        
                        # print("Eval_steps: ",current_lengths[i]," Eval_return: ",current_rewards[i])
                        
                        # if is_monitor_wrapped:
                        #     # Atari wrapper can send a "done" signal when
                        #     # the agent loses a life, but it does not correspond
                        #     # to the true end of episode
                        #     if "episode" in info.keys():
                        #         # Do not trust "done" with episode endings.
                        #         # Monitor wrapper includes "episode" key in info if environment
                        #         # has been wrapped with it. Use those rewards instead.
                        #         episode_rewards.append(info["episode"]["r"])
                        #         episode_lengths.append(info["episode"]["l"])
                        #         # Only increment at the real end of an episode
                        #         episode_counts[i] += 1
                        # else:
                        #     episode_rewards.append(current_rewards[i])
                        #     episode_lengths.append(current_lengths[i])
                        #     episode_counts[i] += 1
                        
                        # episode_rewards.append(current_rewards[i])
                        # episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                        
                        # current_rewards[i] = 0
                        # current_lengths[i] = 0

            if render:
                env.render()

    avg_r = np.mean(reward_data)
    success_rate = np.sum(success_data)/len(success_data)
    avg_t = np.mean(time_data)
    avg_e = np.mean(energy_data)
    
    print(f"++++++++ Evaluation info ++++++++")
    print(f"Avg cumulative reward: {avg_r:.2f}")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Avg time: {avg_t:.2f}")
    print(f"Avg energy: {avg_e:.2f}")
    print(f"++++++++ Evaluation info ++++++++\n")

    # mean_reward = np.mean(episode_rewards)
    # std_reward = np.std(episode_rewards)
    # if reward_threshold is not None:
    #     assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    # if return_episode_rewards:
    #     ##### modification #####
    #     return episode_rewards, episode_lengths, env.episode_data
    # ##### modification #####
    # return mean_reward, std_reward, env.episode_data

    return reward_data, action_data, success_data, time_data, energy_data, env.episode_data
