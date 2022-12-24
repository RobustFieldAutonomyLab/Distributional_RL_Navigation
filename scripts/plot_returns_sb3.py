import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

if __name__ == "__main__":
    first = True
    for seed in [2,3,4,5,6]:
        data_dir = "../experiment_data/experiment_2022-12-23-18-19-03"
        seed_dir = "seed_"+str(seed)
        eval_file = "evaluations.npz"
        evals = np.load(os.path.join(data_dir,seed_dir,eval_file))

        if first:
            timesteps = evals['timesteps']
            returns = evals['results']
            first = False
        else:
            returns = np.hstack((returns,evals['results']))

    # idx = np.argmax(returns)

    # print("max return at ",idx,": ",returns[idx])

    fig, ax = plt.subplots()

    mean_line = np.mean(returns,axis=1)
    yerr = np.std(returns,axis=1) / np.sqrt(np.shape(returns)[1])
    
    ax.plot(timesteps,mean_line,c="b")
    ax.fill_between(timesteps,mean_line+yerr,mean_line-yerr,alpha=0.2)

    plt.show()