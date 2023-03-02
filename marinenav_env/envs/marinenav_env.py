import numpy as np
import scipy.spatial
import marinenav_env.envs.utils.robot as robot
import gym
import json
import copy

class Core:

    def __init__(self, x:float, y:float, clockwise:bool, Gamma:float):

        self.x = x  # x coordinate of the vortex core  
        self.y = y  # y coordinate of the vortex core
        self.clockwise = clockwise # if the vortex direction is clockwise
        self.Gamma = Gamma  # circulation strength of the vortex core

class Obstacle:

    def __init__(self, x:float, y:float, r:float):

        self.x = x # x coordinate of the obstacle center
        self.y = y # y coordinate of the obstacle center
        self.r = r # radius of the obstacle    

class MarineNavEnv(gym.Env):

    def __init__(self, seed:int=0, schedule:dict=None):

        self.robot = robot.Robot()
        self.sd = seed
        self.rd = np.random.RandomState(seed) # PRNG 

        # Define action space and observation space for gym
        self.action_space = gym.spaces.Discrete(self.robot.compute_actions_dimension())
        obs_len = 2 + 2 + 2 * self.robot.sonar.num_beams
        self.observation_space = gym.spaces.Box(low = -np.inf * np.ones(obs_len), \
                                                high = np.inf * np.ones(obs_len), \
                                                dtype = np.float32)
        
        # parameter initialization
        self.width = 50 # x coordinate dimension of the map
        self.height = 50 # y coordinate dimension of the map
        self.r = 0.5  # radius of vortex core
        self.v_rel_max = 1.0 # max allowable speed when two currents flowing towards each other
        self.p = 0.8 # max allowable relative speed at another vortex core
        self.v_range = [5,10] # speed range of the vortex (at the edge of core)
        self.obs_r_range = [1,3] # radius range of the obstacle
        self.clear_r = 10.0 # radius of area centered at start and goal where no vortex cores or obstacles exist
        self.reset_start_and_goal = True # if the start and goal position be set randomly in reset()
        self.start = np.array([5.0,5.0]) # robot start position
        self.random_reset_state = True # if initial state of the robot be set randomly in reset_robot()
        self.init_speed = 0.0 # robot initial forword speed
        self.init_theta = np.pi/4 # robot initial orientation angle
        self.goal = np.array([45.0,45.0]) # goal position
        self.goal_dis = 2.0 # max distance to goal considered as reached
        self.timestep_penalty = -1.0
        # self.dist_reward = self.robot.compute_dist_reward_scale()
        # self.energy_penalty = self.robot.compute_penalty_matrix()
        # self.angle_penalty = -0.5
        self.collision_penalty = -50.0
        self.goal_reward = 100.0
        self.discount = 0.99
        self.num_cores = 8
        self.num_obs = 5
        self.min_start_goal_dis = 25.0

        self.cores = [] # vortex cores
        self.obstacles = [] # cylinder obstacles

        self.schedule = schedule # schedule for curriculum learning
        self.episode_timesteps = 0 # current episode timesteps
        self.total_timesteps = 0 # learning timesteps

        self.set_boundary = False # set boundary of environment

    def get_state_space_dimension(self):
        return 2 + 2 + 2 * self.robot.sonar.num_beams
    
    def get_action_space_dimension(self):
        return self.robot.compute_actions_dimension()

    def reset(self):
        # reset the environment

        if self.schedule is not None:
            steps = self.schedule["timesteps"]
            diffs = np.array(steps) - self.total_timesteps
            
            # find the interval the current timestep falls into
            idx = len(diffs[diffs<=0])-1

            self.num_cores = self.schedule["num_cores"][idx]
            self.num_obs = self.schedule["num_obstacles"][idx]
            self.min_start_goal_dis = self.schedule["min_start_goal_dis"][idx]

            print("======== training env setup ========")
            print("num of cores: ",self.num_cores)
            print("num of obstacles: ",self.num_obs)
            print("min start goal dis: ",self.min_start_goal_dis)
            print("======== training env setup ========\n") 
        
        self.episode_timesteps = 0

        self.cores.clear()
        self.obstacles.clear()

        num_cores = self.num_cores
        num_obs = self.num_obs

        if self.reset_start_and_goal:
        # reset start and goal state randomly
            iteration = 500
            max_dist = 0.0
            while True:
                start = self.rd.uniform(low = 2.0*np.ones(2), high = np.array([self.width-2.0,self.height-2.0]))
                goal = self.rd.uniform(low = 2.0*np.ones(2), high = np.array([self.width-2.0,self.height-2.0]))
                iteration -= 1
                if np.linalg.norm(goal-start) > max_dist:
                    max_dist = np.linalg.norm(goal-start)
                    self.start = start
                    self.goal = goal
                if max_dist > self.min_start_goal_dis or iteration == 0:
                    break

        # generate vortex with random position, spinning direction and strength
        if num_cores > 0:
            iteration = 500
            while True:
                center = self.rd.uniform(low = np.zeros(2), high = np.array([self.width,self.height]))
                direction = self.rd.binomial(1,0.5)
                v_edge = self.rd.uniform(low = self.v_range[0], high = self.v_range[1])
                Gamma = 2 * np.pi * self.r * v_edge
                core = Core(center[0],center[1],direction,Gamma)
                iteration -= 1
                if self.check_core(core):
                    self.cores.append(core)
                    num_cores -= 1
                if iteration == 0 or num_cores == 0:
                    break
        
        centers = None
        for core in self.cores:
            if centers is None:
                centers = np.array([[core.x,core.y]])
            else:
                c = np.array([[core.x,core.y]])
                centers = np.vstack((centers,c))
        
        # KDTree storing vortex core center positions
        if centers is not None:
            self.core_centers = scipy.spatial.KDTree(centers)

        # generate obstacles with random position and size
        if num_obs > 0:
            iteration = 500
            while True:
                center = self.rd.uniform(low = 5.0*np.ones(2), high = np.array([self.width-5.0,self.height-5.0]))
                r = self.rd.uniform(low = self.obs_r_range[0], high = self.obs_r_range[1])
                obs = Obstacle(center[0],center[1],r)
                iteration -= 1
                if self.check_obstacle(obs):
                    self.obstacles.append(obs)
                    num_obs -= 1
                if iteration == 0 or num_obs == 0:
                    break

        centers = None
        for obs in self.obstacles:
            if centers is None:
                centers = np.array([[obs.x,obs.y]])
            else:
                c = np.array([[obs.x,obs.y]])
                centers = np.vstack((centers,c))
        
        # KDTree storing obstacle center positions
        if centers is not None: 
            self.obs_centers = scipy.spatial.KDTree(centers)

        # reset robot state
        self.reset_robot()

        return self.get_observation()

    def reset_robot(self):
        # reset robot state
        if self.random_reset_state:
            self.robot.init_theta = self.rd.uniform(low = 0.0, high = 2*np.pi)
            self.robot.init_speed = self.rd.uniform(low = 0.0, high = self.robot.max_speed)
        else:
            self.robot.init_theta = self.init_theta
            self.robot.init_speed = self.init_speed
        current_v = self.get_velocity(self.start[0],self.start[1])
        self.robot.reset_state(self.start[0],self.start[1], current_velocity=current_v)

    def step(self, action):
        # execute action, update the environment, and return (obs, reward, done)

        # save action to history
        self.robot.action_history.append(action)

        dis_before = self.dist_to_goal()

        # update robot state after executing the action    
        for _ in range(self.robot.N):
            current_velocity = self.get_velocity(self.robot.x, self.robot.y)
            self.robot.update_state(action,current_velocity)
            # save trajectory
            self.robot.trajectory.append([self.robot.x,self.robot.y])

        dis_after = self.dist_to_goal()
        
        # get observation 
        obs = self.get_observation()

        # constant penalty applied at every time step
        reward = self.timestep_penalty

        # # penalize action according to magnitude (energy consumption)
        # a,w = self.robot.actions[action]
        # u = np.matrix([[a],[w]])
        # p = np.transpose(u) * self.energy_penalty * u
        # reward += p[0,0]

        # reward agent for getting closer to the goal
        reward += dis_before-dis_after

        # # penalize agent when the difference of steering direction and velocity direction is too large
        # velocity = obs[:2]
        # diff_angle = 0.0
        # if np.linalg.norm(velocity) > 1e-03:
        #     diff_angle = np.abs(np.arctan2(velocity[1],velocity[0]))

        # if diff_angle > 0.25*self.robot.sonar.angle:
        #     reward += self.angle_penalty * diff_angle

        if self.set_boundary and self.out_of_boundary():
            # No used in training 
            done = True
            info = {"state":"out of boundary"}
        elif self.episode_timesteps >= 1000:
            done = True
            info = {"state":"too long episode"}
        elif self.check_collision():
            reward += self.collision_penalty
            done = True
            info = {"state":"collision"}
        elif self.check_reach_goal():
            reward += self.goal_reward
            done = True
            info = {"state":"reach goal"}
        else:
            done = False
            info = {"state":"normal"}

        self.episode_timesteps += 1
        self.total_timesteps += 1

        return obs, reward, done, info

    def out_of_boundary(self):
        # only used when boundary is set
        x_out = self.robot.x < 0.0 or self.robot.x > self.width
        y_out = self.robot.y < 0.0 or self.robot.y > self.height
        return x_out or y_out

    def dist_to_goal(self):
        return np.linalg.norm(self.goal - np.array([self.robot.x,self.robot.y]))

    def get_observation(self, for_visualize=False):

        # generate observation (1.vehicle velocity wrt seafloor in robot frame by DVL, 
        #                       2.obstacle reflection point clouds in robot frame by Sonar,
        #                       3.goal position in robot frame)
        self.robot.sonar_reflection(self.obstacles)

        # convert information in world frame to robot frame
        R_wr, t_wr = self.robot.get_robot_transform()

        R_rw = np.transpose(R_wr)
        t_rw = -R_rw * t_wr

        # vehicle velocity wrt seafloor in robot frame
        abs_velocity_r = R_rw * np.reshape(self.robot.velocity,(2,1))
        abs_velocity_r.resize((2,))
        abs_velocity_r = np.array(abs_velocity_r)

        # goal position in robot frame
        goal_w = np.reshape(self.goal,(2,1))
        goal_r = R_rw * goal_w + t_rw
        goal_r.resize((2,))
        goal_r = np.array(goal_r)

        # obstacle reflection point clouds in robot frame
        if for_visualize:
            sonar_points_r = None
            for point in self.robot.sonar.reflections:
                p = np.reshape(point,(3,1))
                p[:2] = R_rw * p[:2] + t_rw
                if sonar_points_r is None:
                    sonar_points_r = p
                else:
                    sonar_points_r = np.hstack((sonar_points_r,p))

            return abs_velocity_r, sonar_points_r, goal_r
        else:
            # set virtual points as (0,0), and concatenate all observations 
            # into one vector
            sonar_points_r = None
            for point in self.robot.sonar.reflections:
                p = np.reshape(point,(3,1))
                if p[2] == 0:
                    p_r = np.zeros(2)
                else:
                    p_r = R_rw * p[:2] + t_rw
                    p_r.resize((2,))
                    p_r = np.array(p_r)
                if sonar_points_r is None:
                    sonar_points_r = p_r
                else:
                    sonar_points_r = np.hstack((sonar_points_r,p_r))

            return np.hstack((abs_velocity_r,goal_r,sonar_points_r))


    def check_collision(self):
        if len(self.obstacles) == 0:
            return False
        
        d, idx = self.obs_centers.query(np.array([self.robot.x,self.robot.y]))
        if d <= self.obstacles[idx].r + self.robot.r:
            return True
        return False

    def check_reach_goal(self):
        dis = np.array([self.robot.x,self.robot.y]) - self.goal
        if np.linalg.norm(dis) <= self.goal_dis:
            return True
        return False
    
    def check_core(self,core_j):

        # Within the range of the map
        if core_j.x - self.r < 0.0 or core_j.x + self.r > self.width:
            return False
        if core_j.y - self.r < 0.0 or core_j.y + self.r > self.width:
            return False

        # Not too close to start and goal point
        core_pos = np.array([core_j.x,core_j.y])
        dis_s = core_pos - self.start
        if np.linalg.norm(dis_s) < self.r + self.clear_r:
            return False
        dis_g = core_pos - self.goal
        if np.linalg.norm(dis_g) < self.r + self.clear_r:
            return False

        for core_i in self.cores:
            dx = core_i.x - core_j.x
            dy = core_i.y - core_j.y
            dis = np.sqrt(dx*dx+dy*dy)

            if core_i.clockwise == core_j.clockwise:
                # i and j rotate in the same direction, their currents run towards each other at boundary
                # The currents speed at boundary need to be lower than threshold  
                boundary_i = core_i.Gamma / (2*np.pi*self.v_rel_max)
                boundary_j = core_j.Gamma / (2*np.pi*self.v_rel_max)
                if dis < boundary_i + boundary_j:
                    return False
            else:
                # i and j rotate in the opposite direction, their currents join at boundary
                # The relative current speed of the stronger vortex at boundary need to be lower than threshold 
                Gamma_l = max(core_i.Gamma, core_j.Gamma)
                Gamma_s = min(core_i.Gamma, core_j.Gamma)
                v_1 = Gamma_l / (2*np.pi*(dis-2*self.r))
                v_2 = Gamma_s / (2*np.pi*self.r)
                if v_1 > self.p * v_2:
                    return False

        return True

    def check_obstacle(self,obs):

        # Within the range of the map
        if obs.x - obs.r < 0.0 or obs.x + obs.r > self.width:
            return False
        if obs.y - obs.r < 0.0 or obs.y + obs.r > self.height:
            return False

        # Not too close to start and goal point
        obs_pos = np.array([obs.x,obs.y])
        dis_s = obs_pos - self.start
        if np.linalg.norm(dis_s) < obs.r + self.clear_r:
            return False
        dis_g = obs_pos - self.goal
        if np.linalg.norm(dis_g) < obs.r + self.clear_r:
            return False

        # Not collide with vortex cores
        for core in self.cores:
            dx = core.x - obs.x
            dy = core.y - obs.y
            dis = np.sqrt(dx*dx + dy*dy)

            if dis <= self.r + obs.r:
                return False
        
        # Not collide with other obstacles
        for obstacle in self.obstacles:
            dx = obstacle.x - obs.x
            dy = obstacle.y - obs.y
            dis = np.sqrt(dx*dx + dy*dy)

            if dis <= obstacle.r + obs.r:
                return False
        
        return True

    def get_velocity(self,x:float, y:float):
        if len(self.cores) == 0:
            return np.zeros(2)
        
        # sort the vortices according to their distance to the query point
        d, idx = self.core_centers.query(np.array([x,y]),k=len(self.cores))
        if isinstance(idx,np.int64):
            idx = [idx]

        v_radial_set = []
        v_velocity = np.zeros((2,1))
        for i in list(idx): 
            core = self.cores[i]
            v_radial = np.matrix([[core.x-x],[core.y-y]])

            for v in v_radial_set:
                project = np.transpose(v) * v_radial
                if project[0,0] > 0:
                    # if the core is in the outter area of a checked core (wrt the query position),
                    # assume that it has no influence the velocity of the query position
                    continue
            
            v_radial_set.append(v_radial)
            dis = np.linalg.norm(v_radial)
            v_radial /= dis
            if core.clockwise:
                rotation = np.matrix([[0., -1.],[1., 0]])
            else:
                rotation = np.matrix([[0., 1.],[-1., 0]])
            v_tangent = rotation * v_radial
            speed = self.compute_speed(core.Gamma,dis)
            v_velocity += v_tangent * speed
        
        return np.array([v_velocity[0,0], v_velocity[1,0]])

    def get_velocity_test(self,x:float, y:float):
        v = np.ones(2)
        return v / np.linalg.norm(v)

    def compute_speed(self, Gamma:float, d:float):
        if d <= self.r:
            return Gamma / (2*np.pi*self.r*self.r) * d
        else:
            return Gamma / (2*np.pi*d)

    def reset_with_eval_config(self,eval_config):
        self.episode_timesteps = 0
        
        # load env config
        self.sd = eval_config["env"]["seed"]
        self.width = eval_config["env"]["width"]
        self.height = eval_config["env"]["height"]
        self.r = eval_config["env"]["r"]
        self.v_rel_max = eval_config["env"]["v_rel_max"]
        self.p = eval_config["env"]["p"]
        self.v_range = copy.deepcopy(eval_config["env"]["v_range"])
        self.obs_r_range = copy.deepcopy(eval_config["env"]["obs_r_range"])
        self.clear_r = eval_config["env"]["clear_r"]
        self.start = np.array(eval_config["env"]["start"])
        self.goal = np.array(eval_config["env"]["goal"])
        self.goal_dis = eval_config["env"]["goal_dis"]
        self.timestep_penalty = eval_config["env"]["timestep_penalty"]
        self.collision_penalty = eval_config["env"]["collision_penalty"]
        self.goal_reward = eval_config["env"]["goal_reward"]
        self.discount = eval_config["env"]["discount"]

        # load vortex cores
        self.cores.clear()
        centers = None
        for i in range(len(eval_config["env"]["cores"]["positions"])):
            center = eval_config["env"]["cores"]["positions"][i]
            clockwise = eval_config["env"]["cores"]["clockwise"][i]
            Gamma = eval_config["env"]["cores"]["Gamma"][i]
            core = Core(center[0],center[1],clockwise,Gamma)
            self.cores.append(core)
            if centers is None:
                centers = np.array([[core.x,core.y]])
            else:
                c = np.array([[core.x,core.y]])
                centers = np.vstack((centers,c))
        
        if centers is not None:
            self.core_centers = scipy.spatial.KDTree(centers)

        # load obstacles
        self.obstacles.clear()
        centers = None
        for i in range(len(eval_config["env"]["obstacles"]["positions"])):
            center = eval_config["env"]["obstacles"]["positions"][i]
            r = eval_config["env"]["obstacles"]["r"][i]
            obs = Obstacle(center[0],center[1],r)
            self.obstacles.append(obs)
            if centers is None:
                centers = np.array([[obs.x,obs.y]])
            else:
                c = np.array([[obs.x,obs.y]])
                centers = np.vstack((centers,c))

        if centers is not None:
            self.obs_centers = scipy.spatial.KDTree(centers)

        # load robot config
        self.robot.dt = eval_config["robot"]["dt"]
        self.robot.N = eval_config["robot"]["N"]
        self.robot.length = eval_config["robot"]["length"]
        self.robot.width = eval_config["robot"]["width"]
        self.robot.r = eval_config["robot"]["r"]
        self.robot.max_speed = eval_config["robot"]["max_speed"]
        self.robot.a = np.array(eval_config["robot"]["a"])
        self.robot.w = np.array(eval_config["robot"]["w"])
        self.robot.compute_k()
        self.robot.compute_actions()
        self.robot.init_theta = eval_config["robot"]["init_theta"]
        self.robot.init_speed = eval_config["robot"]["init_speed"]

        # load sonar config
        self.robot.sonar.range = eval_config["robot"]["sonar"]["range"]
        self.robot.sonar.angle = eval_config["robot"]["sonar"]["angle"]
        self.robot.sonar.num_beams = eval_config["robot"]["sonar"]["num_beams"]
        self.robot.sonar.compute_phi()
        self.robot.sonar.compute_beam_angles()

        # update env action and observation space
        self.action_space = gym.spaces.Discrete(self.robot.compute_actions_dimension())
        obs_len = 2 + 2 + 2 * self.robot.sonar.num_beams
        self.observation_space = gym.spaces.Box(low = -np.inf * np.ones(obs_len), \
                                                    high = np.inf * np.ones(obs_len), \
                                                    dtype = np.float32)

        # reset robot state
        current_v = self.get_velocity(self.start[0],self.start[1])
        self.robot.reset_state(self.start[0],self.start[1], current_velocity=current_v)

        return self.get_observation()          

    def episode_data(self):
        episode = {}

        # save environment config
        episode["env"] = {}
        episode["env"]["seed"] = self.sd
        episode["env"]["width"] = self.width
        episode["env"]["height"] = self.height
        episode["env"]["r"] = self.r
        episode["env"]["v_rel_max"] = self.v_rel_max
        episode["env"]["p"] = self.p
        episode["env"]["v_range"] = copy.deepcopy(self.v_range) 
        episode["env"]["obs_r_range"] = copy.deepcopy(self.obs_r_range)
        episode["env"]["clear_r"] = self.clear_r
        episode["env"]["start"] = list(self.start)
        episode["env"]["goal"] = list(self.goal)
        episode["env"]["goal_dis"] = self.goal_dis
        episode["env"]["timestep_penalty"] = self.timestep_penalty
        # episode["env"]["energy_penalty"] = self.energy_penalty.tolist()
        # episode["env"]["angle_penalty"] = self.angle_penalty
        episode["env"]["collision_penalty"] = self.collision_penalty
        episode["env"]["goal_reward"] = self.goal_reward
        episode["env"]["discount"] = self.discount

        # save vortex cores information
        episode["env"]["cores"] = {}
        episode["env"]["cores"]["positions"] = []
        episode["env"]["cores"]["clockwise"] = []
        episode["env"]["cores"]["Gamma"] = []
        for core in self.cores:
            episode["env"]["cores"]["positions"].append([core.x,core.y])
            episode["env"]["cores"]["clockwise"].append(core.clockwise)
            episode["env"]["cores"]["Gamma"].append(core.Gamma)

        # save obstacles information
        episode["env"]["obstacles"] = {}
        episode["env"]["obstacles"]["positions"] = []
        episode["env"]["obstacles"]["r"] = []
        for obs in self.obstacles:
            episode["env"]["obstacles"]["positions"].append([obs.x,obs.y])
            episode["env"]["obstacles"]["r"].append(obs.r)

        # save robot config
        episode["robot"] = {}
        episode["robot"]["dt"] = self.robot.dt
        episode["robot"]["N"] = self.robot.N
        episode["robot"]["length"] = self.robot.length
        episode["robot"]["width"] = self.robot.width
        episode["robot"]["r"] = self.robot.r
        episode["robot"]["max_speed"] = self.robot.max_speed
        episode["robot"]["a"] = list(self.robot.a)
        episode["robot"]["w"] = list(self.robot.w)
        episode["robot"]["init_theta"] = self.robot.init_theta
        episode["robot"]["init_speed"] = self.robot.init_speed

        # save sonar config
        episode["robot"]["sonar"] = {}
        episode["robot"]["sonar"]["range"] = self.robot.sonar.range
        episode["robot"]["sonar"]["angle"] = self.robot.sonar.angle
        episode["robot"]["sonar"]["num_beams"] = self.robot.sonar.num_beams

        # save action history
        episode["robot"]["action_history"] = copy.deepcopy(self.robot.action_history)
        episode["robot"]["trajectory"] = copy.deepcopy(self.robot.trajectory)

        return episode

    def save_episode(self,filename):
        episode = self.episode_data()
        with open(filename,"w") as file:
            json.dump(episode,file)
