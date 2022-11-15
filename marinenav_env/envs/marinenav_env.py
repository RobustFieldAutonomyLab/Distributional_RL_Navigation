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

    def __init__(self, seed:int=0):

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
        self.clear_r = 5 # radius of area centered at start and goal where no vortex cores or obstacles exist
        self.start = np.array([5.0,5.0]) # robot start position
        self.goal = np.array([45.0,45.0]) # goal position
        self.goal_dis = 2.0 # max distance to goal considered as reached
        self.timestep_penalty = -1.0
        self.energy_penalty = self.robot.compute_penalty_matrix()
        self.collision_penalty = -50.0
        self.goal_reward = 100.0
        self.discount = 0.99
        self.num_cores = 8
        self.num_obs = 5

        self.cores = [] # vortex cores
        self.obstacles = [] # cylinder obstacles

    def reset(self):
        # reset the environment
        
        self.cores.clear()
        self.obstacles.clear()

        num_cores = self.num_cores
        num_obs = self.num_obs

        # generate vortex with random position, spinning direction and strength
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
        self.core_centers = scipy.spatial.KDTree(centers)

        # generate obstacles with random position and size
        iteration = 500
        while True:
            center = self.rd.uniform(low = np.zeros(2), high = np.array([self.width,self.height]))
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
        self.obs_centers = scipy.spatial.KDTree(centers)

        # reset robot state
        self.reset_robot()

        return self.get_observation()

    def reset_robot(self):
        # reset robot state
        current_v = self.get_velocity(self.start[0],self.start[1])
        self.robot.reset_state(self.start[0],self.start[1],current_velocity=current_v)

    def step(self, action):
        # execute action, update the environment, and return (obs, reward, done)

        # save action to history
        self.robot.action_history.append(action)

        # update robot state after executing the action    
        for _ in range(self.robot.N):
            current_velocity = self.get_velocity(self.robot.x, self.robot.y)
            self.robot.update_state(action,current_velocity)
        
        # get observation 
        obs = self.get_observation()

        # constant penalty applied at every time step
        reward = self.timestep_penalty

        # penalize action according to magnitude (energy consumption)
        a,w = self.robot.actions[action]
        u = np.matrix([[a],[w]])
        p = np.transpose(u) * self.energy_penalty * u
        reward += p[0,0]

        if self.check_collision():
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

        return obs, reward, done, info

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
        # sort the vortices according to their distance to the query point
        d, idx = self.core_centers.query(np.array([x,y]),k=len(self.cores))

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

    def compute_speed(self, Gamma:float, d:float):
        if d <= self.r:
            return Gamma / (2*np.pi*self.r*self.r) * d
        else:
            return Gamma / (2*np.pi*d)          

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
        episode["env"]["energy_penalty"] = self.energy_penalty.tolist()
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

        # save sonar config
        episode["robot"]["sonar"] = {}
        episode["robot"]["sonar"]["range"] = self.robot.sonar.range
        episode["robot"]["sonar"]["angle"] = self.robot.sonar.angle
        episode["robot"]["sonar"]["num_beams"] = self.robot.sonar.num_beams

        # save action history
        episode["robot"]["action_history"] = copy.deepcopy(self.robot.action_history)

        return episode

    def save_episode(self,filename):
        episode = self.episode_data()
        with open(filename,"w") as file:
            json.dump(episode,file)
