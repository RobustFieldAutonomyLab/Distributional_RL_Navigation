import torch
import torch.optim as optim
import numpy as np
import random
from thirdparty.IQN.model import IQN
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
        self.qnetwork_local = IQN(self.state_size, self.action_size, layer_size, seed, device).to(device)
        self.qnetwork_target = IQN(self.state_size, self.action_size, layer_size, seed, device).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)
        #print(self.qnetwork_local)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, seed, GAMMA, n_step)
        
        # Current time step
        self.current_timestep = 0

        # Learning time step (start counting after learning_starts time step)
        self.learning_timestep = 0

        # Evaluation data
        self.eval_timesteps = []
        self.eval_ep_rewards = []
        self.eval_ep_lengths = []
        self.eval_ep_data = []

    def load_model(self,path,device="cpu"):
        # load trained IQN models
        self.qnetwork_local = IQN.load(path,device)
        self.qnetwork_target = IQN.load(path,device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

    def learn(self,
              total_timesteps,
              train_env,
              eval_env,
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
            next_state, reward, done, _ = train_env.step(action)

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
                    # evaluate at risk neutral level 
                    self.evaluation(eval_env,eval_log_path=eval_log_path)

                self.learning_timestep += 1

            if done:
                ep_num += 1
                
                if verbose:
                    # print abstract info of learning process
                    print("======== training info ========")
                    print("current ep_length: ",ep_length)
                    print("current ep_reward: ",ep_reward)
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

    def act_eval_IQN(self, state, eps, cvar=1.0):
        """Returns action return quantiles and action decision
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

    def evaluation(self,eval_env,cvar=1.0,eval_log_path=None):
        """Evaluate performance of the agent
        Params
        ======
            eval_env (gym compatible env): evaluation environment
            cvar (float): CVaR value
        """
        observation = eval_env.reset()
        cumulative_reward = 0.0
        length = 0
        done = False
        
        while not done and length < 1000:
            action = self.act(observation,0.0,cvar)
            observation, reward, done, _ = eval_env.step(action)
            cumulative_reward += eval_env.discount ** length * reward
            length += 1
            # if length % 50 == 0:
            #     print(length)

        print("++++++++ Evaluation info ++++++++")
        print("Episode length: ",length)
        print("Cumulative reward: ",cumulative_reward)
        print("++++++++ Evaluation info ++++++++\n")

        self.eval_timesteps.append(self.current_timestep)
        self.eval_ep_rewards.append(cumulative_reward)
        self.eval_ep_lengths.append(length)
        self.eval_ep_data.append(eval_env.episode_data())

        if eval_log_path is not None:
            # save evaluation data
            np.savez(
                os.path.join(eval_log_path,"evaluations.npz"),
                timesteps=self.eval_timesteps,
                episode_rewards=self.eval_ep_rewards,
                episode_lengths=self.eval_ep_lengths,
                episode_data=self.eval_ep_data
            )

            # save the latest IQN model
            self.qnetwork_local.save(eval_log_path)


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss