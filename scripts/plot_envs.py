import sys
sys.path.insert(0,"../")
import env_visualizer
import json
import numpy as np

if __name__ == "__main__":
    ev = env_visualizer.EnvVisualizer(seed=9,draw_envs=True)

    eval_env = ev.env
    
    env_configs = []

    # varying the number of cores and obstacles to adjust the level of difficulty
    num_cs = [4,6,8]
    num_os = [6,8,10]
    min_start_goal_dis=[30.0,35.0,40.0]

    for i in range(len(num_cs)):
        eval_env.num_cores = num_cs[i]
        eval_env.num_obs = num_os[i]
        eval_env.min_start_goal_dis = min_start_goal_dis[i]

        eval_env.reset()

        # save eval config
        env_configs.append(eval_env.episode_data())
    

    ev.init_visualize(env_configs)
    

    ev.fig.tight_layout()
    ev.fig.savefig("training_envs_test.png")