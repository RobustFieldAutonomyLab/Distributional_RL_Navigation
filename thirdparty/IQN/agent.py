import torch
import torch.optim as optim
import numpy as np
import random
from thirdparty.IQN.model import IQN
from thirdparty.IQN.model import ObsEncoder
from thirdparty.IQN.replay_buffer import ReplayBuffer
import os

class IQNAgent():
    """Interacts with and learns from the environment."""
    def __init__(self, 
                 state_size, 
                 action_size,
                 layer_size=64, 
                 n_step=1, 
                 BATCH_SIZE=32, 
                 BUFFER_SIZE=1_000_000,
                 LR=1e-4, 
                 TAU=1.0, 
                 GAMMA=0.99, 
                 UPDATE_EVERY=4,
                 learning_starts=10000,
                 target_update_interval=10000,
                 exploration_fraction=0.1,
                 initial_eps=1.0,
                 final_eps=0.05, 
                 device="cpu", 
                 seed=0):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.LR = LR
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.n_step = n_step
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.initial_eps = initial_eps
        self.final_eps = final_eps

        # IQN-Network
        # self.qnetwork_local = IQN(self.state_size, self.action_size, layer_size, seed, device).to(device)
        # self.qnetwork_target = IQN(self.state_size, self.action_size, layer_size, seed, device).to(device)
        self.qnetwork_local = ObsEncoder(self.state_size, self.action_size, seed, device).to(device)
        self.qnetwork_target = ObsEncoder(self.state_size, self.action_size, seed, device).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)
        #print(self.qnetwork_local)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, seed, GAMMA, n_step)
        
        # Current time step
        self.current_timestep = 0

        # Learning time step (start counting after learning_starts time step)
        self.learning_timestep = 0

        # Evaluation data
        self.eval_timesteps = dict(greedy=[],adaptive=[])
        self.eval_actions = dict(greedy=[],adaptive=[])
        self.eval_rewards = dict(greedy=[],adaptive=[])
        self.eval_successes = dict(greedy=[],adaptive=[])
        self.eval_times = dict(greedy=[],adaptive=[])
        self.eval_energies = dict(greedy=[],adaptive=[])

    def load_model(self,path,device="cpu"):
        # load trained IQN models
        # self.qnetwork_local = IQN.load(path,device)
        # self.qnetwork_target = IQN.load(path,device)
        self.qnetwork_local = ObsEncoder.load(path,device)
        self.qnetwork_target = ObsEncoder.load(path,device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

    def learn(self,
              total_timesteps,
              train_env,
              eval_env,
              eval_config,
              eval_freq,
              eval_log_path,
              verbose=True):
        
        state = train_env.reset()

        # # Sample CVaR value from (0.0,1.0)
        # cvar = 1 - np.random.uniform(0.0, 1.0)

        # current episode 
        ep_reward = 0.0
        ep_length = 0
        ep_num = 0
        
        while self.current_timestep <= total_timesteps:
            eps = self.linear_eps(total_timesteps)
            action = self.act(state,eps)
            next_state, reward, done, info = train_env.step(action)

            ep_reward += train_env.discount ** ep_length * reward
            ep_length += 1
            
            # Save experience in replay memory
            self.memory.add(state, action, reward, next_state, done)

            state = next_state
            
            # Learn, update and evaluate models after learning_starts time step 
            if self.current_timestep >= self.learning_starts:
                # Learn every UPDATE_EVERY time steps.
                if self.learning_timestep % self.UPDATE_EVERY == 0:
                    # If enough samples are available in memory, get random subset and learn
                    if len(self.memory) > self.BATCH_SIZE:
                        experiences = self.memory.sample()
                        self.train(experiences)

                # Update the target model every target_update_interval time steps
                if self.learning_timestep % self.target_update_interval == 0:
                    self.soft_update(self.qnetwork_local, self.qnetwork_target)

                # Evaluate the target model every eval_freq time steps
                if self.learning_timestep % eval_freq == 0:
                    # 1. Evaluate greedy policy (CVaR = 1.0)  
                    self.evaluation(eval_env,eval_config=eval_config,eval_log_path=eval_log_path)
                    
                    # 2. Evaluate adpative CVaR policy
                    self.evaluation(eval_env,eval_config=eval_config,greedy=False,eval_log_path=eval_log_path)

                    # save the latest IQN model
                    self.qnetwork_local.save(eval_log_path)

                self.learning_timestep += 1

            if done:
                ep_num += 1
                
                if verbose:
                    # print abstract info of learning process
                    print("======== training info ========")
                    print("current ep_length: ",ep_length)
                    print("current ep_reward: ",ep_reward)
                    print("current ep_result: ",info["state"])
                    print("episodes_num: ",ep_num)
                    print("exploration_rate: ",eps)
                    print("current_timesteps: ",self.current_timestep)
                    print("total_timesteps: ",total_timesteps)
                    print("======== training info ========\n") 
                
                ep_reward = 0.0
                ep_length = 0

                state = train_env.reset()
                # cvar = 1 - np.random.uniform(0.0, 1.0)

            self.current_timestep += 1


    def linear_eps(self,total_timesteps):
        
        progress = self.current_timestep / total_timesteps
        if progress < self.exploration_fraction:
            r = progress / self.exploration_fraction
            return self.initial_eps + r * (self.final_eps - self.initial_eps)
        else:
            return self.final_eps


    def act(self, state, eps, cvar=1.0):
        """Returns action indexes for given state as per current policy.
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.get_qvals(state, cvar)
        self.qnetwork_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        
        return action

    def act_adaptive(self,state,eps):
        """adptively tune the CVaR value, compute action index and quantiles
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
        """
        cvar = self.adjust_cvar(state)
        return self.act(state,eps,cvar), cvar

    def act_eval(self, state, eps=0.0, cvar=1.0):
        """Returns action index and quantiles 
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            quantiles, taus = self.qnetwork_local.forward(state, self.qnetwork_local.K, cvar)
            action_values = quantiles.mean(dim=1)
        self.qnetwork_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        
        return action, quantiles.cpu().data.numpy(), taus.cpu().data.numpy()

    def act_adaptive_eval(self, state, eps=0.0):
        """adptively tune the CVaR value, compute action index and quantiles
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
        """
        cvar = self.adjust_cvar(state)
        return self.act_eval(state, eps, cvar), cvar

    def adjust_cvar(self,state):
        # scale CVaR value according to the closest distance to obstacles
        sonar_points = state[4:]
        
        closest_d = np.inf
        for i in range(0,len(sonar_points),2):
            x = sonar_points[i]
            y = sonar_points[i+1]

            if np.abs(x) < 1e-3 and np.abs(y) < 1e-3:
                continue

            closest_d = min(closest_d, np.linalg.norm(sonar_points[i:i+2]))
        
        cvar = 1.0
        if closest_d < 10.0:
            cvar = closest_d / 10.0

        return cvar

    def train(self, experiences):
        """Update value parameters using given batch of experience tuples
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next, _ = self.qnetwork_target(next_states)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1) # (batch_size, 1, N)
        
        # Compute Q targets for current states 
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA ** self.n_step * Q_targets_next * (1. - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected, taus = self.qnetwork_local(states)
        Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, 8, 8), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
        
        loss = quantil_l.sum(dim=1).mean(dim=1) # keepdim=True if per weights get multiple
        loss = loss.mean()

        # minimize the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 0.5)
        self.optimizer.step()

        # # update target network
        # self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

    def evaluation(self,
                   eval_env,
                   eval_config,
                   greedy=True,
                   eval_log_path=None):
        """Evaluate performance of the agent
        Params
        ======
            eval_env (gym compatible env): evaluation environment
            eval_config: eval envs config file
        """
        action_data = []
        reward_data = []
        success_data = []
        time_data = []
        energy_data = []
        
        for idx, config in enumerate(eval_config.values()):
            print(f"Evaluating episode {idx}")
            observation = eval_env.reset_with_eval_config(config)
            actions = []
            cumulative_reward = 0.0
            length = 0
            energy = 0.0
            done = False
            
            while not done and length < 1000:
                if greedy:
                    action = self.act(observation,eps=0.0)
                else:
                    action,_ = self.act_adaptive(observation,eps=0.0)    
                observation, reward, done, info = eval_env.step(action)
                cumulative_reward += eval_env.discount ** length * reward
                length += 1
                energy += eval_env.robot.compute_action_energy_cost(int(action))
                actions.append(int(action))

            success = True if info["state"] == "reach goal" else False
            time = eval_env.robot.dt * eval_env.robot.N * length

            action_data.append(actions)
            reward_data.append(cumulative_reward)
            success_data.append(success)
            time_data.append(time)
            energy_data.append(energy)
        
        avg_r = np.mean(reward_data)
        success_rate = np.sum(success_data)/len(success_data)
        idx = np.where(np.array(success_data) == 1)[0]
        avg_t = np.mean(np.array(time_data)[idx])
        avg_e = np.mean(np.array(energy_data)[idx])

        policy = "greedy" if greedy else "adaptive"
        print(f"++++++++ Evaluation info ({policy} IQN) ++++++++")
        print(f"Avg cumulative reward: {avg_r:.2f}")
        print(f"Success rate: {success_rate:.2f}")
        print(f"Avg time: {avg_t:.2f}")
        print(f"Avg energy: {avg_e:.2f}")
        print(f"++++++++ Evaluation info ({policy} IQN) ++++++++\n")

        self.eval_timesteps[policy].append(self.current_timestep)
        self.eval_actions[policy].append(action_data)
        self.eval_rewards[policy].append(reward_data)
        self.eval_successes[policy].append(success_data)
        self.eval_times[policy].append(time_data)
        self.eval_energies[policy].append(energy_data)

        if eval_log_path is not None:
            filename = "greedy_evaluations.npz" if greedy else "adaptive_evaluations.npz"
            
            # save evaluation data
            np.savez(
                os.path.join(eval_log_path,filename),
                timesteps=self.eval_timesteps[policy],
                actions=self.eval_actions[policy],
                rewards=self.eval_rewards[policy],
                successes=self.eval_successes[policy],
                times=self.eval_times[policy],
                energies=self.eval_energies[policy]
            )


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss