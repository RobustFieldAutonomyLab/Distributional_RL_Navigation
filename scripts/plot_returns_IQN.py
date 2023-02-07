import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os

if __name__ == "__main__":
    first = True
    seed = 2
    seed_dir = "seed_"+str(seed)
    eval_agents = ["IQN","DQN"]
    colors = ["b","tab:orange"]

    fig, (ax_rewards,ax_success_rate) = plt.subplots(2,1,figsize=(8,8))
    
    for idx, eval_agent in enumerate(eval_agents):
        if eval_agent == "IQN":
            data_dir = "../training_data/training_2023-02-02-23-24-30" # IQN
            eval_data = np.load(os.path.join(data_dir,seed_dir,f"greedy_evaluations.npz")) # IQN
        else:
            data_dir = "../training_data/training_2023-02-02-23-27-17" # DQN
            eval_data = np.load(os.path.join(data_dir,seed_dir,"evaluations.npz")) # DQN

        timesteps = []
        rewards = []
        success_rates = []
        times = []
        energies = []
        for i in range(len(eval_data['timesteps'])):
            timesteps.append(eval_data['timesteps'][i])
            rewards.append(eval_data['rewards'][i])
            successes = eval_data['successes'][i]
            success_rates.append(np.sum(successes)/len(successes))
            times.append(eval_data['times'][i])
            energies.append(eval_data['energies'][i])
        
        mpl.rcParams["font.size"]=16
        ax_rewards.tick_params(axis="x", labelsize=14)
        ax_rewards.tick_params(axis="y", labelsize=14)
        ax_rewards.plot(timesteps,np.mean(rewards,axis=1),linewidth=2,label=eval_agent,c=colors[idx],zorder=5-idx)
        ax_rewards.xaxis.set_ticks(np.arange(0,eval_data['timesteps'][-1]+1,200000))
        scale_x = 1e-5
        ticks_x = ticker.FuncFormatter(lambda x, pos:'{0:g}'.format(x*scale_x))
        ax_rewards.xaxis.set_major_formatter(ticks_x)
        # ax_rewards.set_xlabel("Timestep(x10^5)",fontsize=15)
        ax_rewards.set_ylabel("Cumulative Reward",fontsize=15)
        ax_rewards.legend(loc="lower right",bbox_to_anchor=(1, 0.2))

        ax_success_rate.tick_params(axis="x", labelsize=14)
        ax_success_rate.tick_params(axis="y", labelsize=14)
        ax_success_rate.plot(timesteps,success_rates,linewidth=2,label=eval_agent,c=colors[idx],zorder=5-idx)
        ax_success_rate.xaxis.set_ticks(np.arange(0,eval_data['timesteps'][-1]+1,200000))
        scale_x = 1e-5
        ticks_x = ticker.FuncFormatter(lambda x, pos:'{0:g}'.format(x*scale_x))
        ax_success_rate.xaxis.set_major_formatter(ticks_x)
        ax_success_rate.set_xlabel("Timestep(x10^5)",fontsize=15)
        ax_success_rate.yaxis.set_ticks(np.arange(0,1.1,0.2))
        ax_success_rate.set_ylabel("Success Rate",fontsize=15)
        ax_success_rate.legend(loc="lower right",bbox_to_anchor=(1, 0.2))

        # ax_times.plot(timesteps,np.mean(times,axis=1))
        # ax_energies.plot(timesteps,np.mean(energies,axis=1))

    fig.tight_layout()
    fig.savefig("learning_curves.png")