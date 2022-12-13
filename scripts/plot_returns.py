import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

if __name__ == "__main__":
    # data_path = "../experiment_12_04"
    data_path = "../experiment_first"
    eval_file = "evaluations.npz"
    evals = np.load(os.path.join(data_path,eval_file))

    timesteps = evals['timesteps']
    returns = evals['results']

    idx = np.argmax(returns)

    print("max return at ",idx,": ",returns[idx])

    fig, ax = plt.subplots()

    ax.plot(timesteps,returns)

    plt.show()