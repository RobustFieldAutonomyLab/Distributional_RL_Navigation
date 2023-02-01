import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

if __name__ == "__main__":
    first = True
    seed = 3
    # data_dir = "../training_data/training_2023-01-31-21-57-02" # IQN
    data_dir = "../training_data/training_2023-01-31-22-24-24" # DQN
    seed_dir = "seed_"+str(seed)
    eval_agent = "greedy"

    fig, ((ax_rewards,ax_success_rate),(ax_times,ax_energies)) = plt.subplots(2,2)
    
    # eval_data = np.load(os.path.join(data_dir,seed_dir,f"{eval_agent}_evaluations.npz")) # IQN
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

    ax_rewards.plot(timesteps,np.mean(rewards,axis=1))
    ax_success_rate.plot(timesteps,success_rates)
    ax_times.plot(timesteps,np.mean(times,axis=1))
    ax_energies.plot(timesteps,np.mean(energies,axis=1))

    plt.show()