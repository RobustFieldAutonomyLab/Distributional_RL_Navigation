import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

##### local modification #####
def ssd_policy(quantiles:np.ndarray, use_threshold:bool=False, mean_threshold:float=1e-03):
    means = np.mean(quantiles,axis=0)
    sort_idx = np.argsort(-1*means)
    best_1 = sort_idx[0]
    best_2 = sort_idx[1]
    if means[best_1] - means[best_2] > mean_threshold:
        return best_1
    else:
        if use_threshold:
            signed_second_moment = -1 * np.var(quantiles,axis=0)
        else:
            signed_second_moment = -1 * np.mean(quantiles**2,axis=0)
        action = best_1
        if signed_second_moment[best_2] > signed_second_moment[best_1]:
            action = best_2
        return action


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    ##### local modification #####
    eval_policy: str = "Greedy",
    ssd_thres: float = 1e-03
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
    
    ##### local modification #####
    # store quantiles prediction for all state action pair if the agent is QR-DQN
    if env.save_q_vals:
        print("saving quantiles (QR-DQN)")
        all_quantiles = []
        for i in range(env.num_states):
            obs = env.get_obs_at_state(i)
            q_vals = model.predict_quantiles(obs)
            all_quantiles.append(q_vals.cpu().data.numpy()[0])
        
        env.save_quantiles(np.array(all_quantiles))

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    while (episode_counts < episode_count_targets).any():
        ##### local modification #####
        if eval_policy == "Greedy":
            actions, states = model.predict(observations, state=states, deterministic=deterministic)
        # TODO: consider multi environments case
        elif eval_policy == "SSD":
            q_vals = model.predict_quantiles(observations)
            actions = np.array([ssd_policy(q_vals.cpu().data.numpy()[0])])
            states = None
        elif eval_policy == "Thresholded_SSD":
            q_vals = model.predict_quantiles(observations)
            actions = np.array([ssd_policy(q_vals.cpu().data.numpy()[0],use_threshold=True,mean_threshold=ssd_thres)])
            states = None
        else:
            raise RuntimeError("The evaluation policy is not available.")
        
        observations, rewards, dones, infos = env.step(actions)
        ##### local modification #####
        #current_rewards += rewards
        current_rewards += env.discount ** current_lengths[0] * rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                ##### local modification #####
                # if dones[i]:
                if dones[i] or current_lengths[i] >= 1000:
                    print("Eval_steps: ",current_lengths[i]," Eval_return: ",current_rewards[i])
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

                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1

                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
