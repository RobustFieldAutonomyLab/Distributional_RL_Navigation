import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

if __name__ == "__main__":
    first = True
    for seed in [2,3,4,5,6]:
        data_dir = "../experiment_data/experiment_2022-12-23-18-02-05"
        seed_dir = "seed_"+str(seed)
        eval_file = "evaluations.npz"
        evals = np.load(os.path.join(data_dir,seed_dir,eval_file))

        if first:
            timesteps = evals['timesteps']
            returns = evals['episode_rewards']
            first = False
        else:
            returns = np.vstack((returns,evals['episode_rewards']))

    fig, ax = plt.subplots()

    mean_line = np.mean(returns,axis=0)
    yerr = np.std(returns,axis=0) / np.sqrt(np.shape(returns)[0])
    
    ax.plot(timesteps,mean_line,c="b")
    ax.fill_between(timesteps,mean_line+yerr,mean_line-yerr,alpha=0.2)

    plt.show()