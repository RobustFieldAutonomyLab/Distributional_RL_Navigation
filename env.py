from matplotlib.animation import Animation
import numpy as np
import scipy.spatial
import matplotlib as mpl
import matplotlib.pyplot as plt
import robot
import gym

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

class Env(gym.Env):

    def __init__(self, seed:int=0):
        
        # parameter initialization
        self.robot = robot.Robot()
        self.rd = np.random.RandomState(seed) # PRNG 
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
        self.num_cores = 8
        self.num_obs = 5

        self.cores = [] # vortex cores
        self.obstacles = [] # cylinder obstacles

        # visualization
        self.fig = None # figure for visualization
        self.axis_graph = None # sub figure for the map
        self.robot_sec = None
        self.robot_last_pos = None
        self.sonar_sec = []
        self.axis_sonar = None # sub figure for Sonar measurement
        self.axis_dvl = None # sub figure for DVL measurement

        self.reset()

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
        current_v = self.get_velocity(self.start[0],self.start[1])
        self.robot.set_state(self.start[0],self.start[1],current_velocity=current_v)

    def step(self, action):
        # execute action (linear acceleration, angular velocity), update the environment, and return (obs, reward, done)

        # update robot state after executing the action    
        for _ in range(self.robot.N):
            current_velocity = self.get_velocity(self.robot.x, self.robot.y)
            self.robot.update_state(action,current_velocity)
        
        # get observation 
        obs = self.get_observation()

        # constant penalty applied at every time step
        reward = self.timestep_penalty

        # penalize action according to magnitude (energy consumption)
        a = self.robot.a(action[0])
        w = self.robot.w(action[1])
        u = np.matrix([[a],[w]])
        p = np.transpose(u) * self.energy_penalty * u
        reward += p[0,0]

        if self.check_collision():
            reward += self.collision_penalty
            done = True
            info = "collision"
        elif self.check_reach_goal():
            reward += self.goal_reward
            done = True
            info = "reach goal"
        else:
            done = False
            info = "normal"

        return obs, reward, done, info

    def get_observation(self, for_plotting=False):

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

        # goal position in robot frame
        goal_w = np.reshape(self.goal,(2,1))
        goal_r = R_rw * goal_w + t_rw
        goal_r.resize((2,))

        # obstacle reflection point clouds in robot frame
        if for_plotting:
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
                if sonar_points_r is None:
                    sonar_points_r = p
                else:
                    sonar_points_r = np.hstack((sonar_points_r,p))

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

    def init_visualize(self):
        self.robot_last_pos = None

        x_pos = list(np.linspace(0,self.width,100))
        y_pos = list(np.linspace(0,self.height,100))

        pos_x = []
        pos_y = []
        arrow_x = []
        arrow_y = []
        for x in x_pos:
            for y in y_pos:
                v = self.get_velocity(x,y)
                pos_x.append(x)
                pos_y.append(y)
                arrow_x.append(v[0])
                arrow_y.append(v[1])
        
        # initialize subplot for the map, robot state and sensor measurments
        self.fig = plt.figure(figsize=(24,16))
        spec = self.fig.add_gridspec(2,3)
        self.axis_graph = self.fig.add_subplot(spec[:,:2])
        self.axis_sonar = self.fig.add_subplot(spec[0,2])
        self.axis_dvl = self.fig.add_subplot(spec[1,2])
        
        # plot current velocity in the map
        self.axis_graph.quiver(pos_x, pos_y, arrow_x, arrow_y)

        # plot obstacles in the map
        for obs in self.obstacles:
            self.axis_graph.add_patch(mpl.patches.Circle((obs.x,obs.y),radius=obs.r))

        self.axis_graph.set_aspect('equal')
        self.axis_graph.set_xlim([0.0,self.width])
        self.axis_graph.set_ylim([0.0,self.height])
    
    def plot_robot(self):
        if self.robot_sec != None:
            self.robot_sec.remove()

        d = np.matrix([[0.5*self.robot.length],[0.5*self.robot.width]])
        rot = np.matrix([[np.cos(self.robot.theta),-np.sin(self.robot.theta)], \
                         [np.sin(self.robot.theta),np.cos(self.robot.theta)]])
        d_r = rot * d
        xy = (self.robot.x-d_r[0,0],self.robot.y-d_r[1,0])

        angle_d = self.robot.theta / np.pi * 180
        self.robot_sec = self.axis_graph.add_patch(mpl.patches.Rectangle(xy,self.robot.length, \
                                                   self.robot.width,     \
                                                   color='g',angle=angle_d,zorder=6))

        if self.robot_last_pos != None:
            self.axis_graph.plot((self.robot_last_pos[0],self.robot.x),
                                 (self.robot_last_pos[1],self.robot.y),
                                 color='m')
        
        self.robot_last_pos = [self.robot.x, self.robot.y]

    def plot_measurements(self):
        self.axis_sonar.clear()
        self.axis_dvl.clear()
        for plot in self.sonar_sec:
            plot[0].remove()
        self.sonar_sec.clear()
        
        abs_velocity_r, sonar_points_r, goal_r = self.get_observation(for_plotting=True)
        
        # plot Sonar beams in the world frame
        for point in self.robot.sonar.reflections:
            x = point[0]
            y = point[1]
            if point[-1] == 0:
                # compute beam range end point 
                x = self.robot.x + 0.5 * (x-self.robot.x)
                y = self.robot.y + 0.5 * (y-self.robot.y)
            else:
                # mark the reflection point
                self.sonar_sec.append(self.axis_graph.plot(x,y,'rx'))

            self.sonar_sec.append(self.axis_graph.plot([self.robot.x,x],[self.robot.y,y],'r--'))

        # plot Sonar reflections in the robot frame (rotate x-axis by 90 degree (upward) in the plot)
        low_angle = np.pi/2 + self.robot.sonar.beam_angles[0]
        high_angle = np.pi/2 + self.robot.sonar.beam_angles[-1]
        low_angle_d = low_angle / np.pi * 180
        high_angle_d = high_angle / np.pi * 180
        self.axis_sonar.add_patch(mpl.patches.Wedge((0.0,0.0),self.robot.sonar.range, \
                                               low_angle_d,high_angle_d,color="r",alpha=0.2))
        
        for i in range(np.shape(sonar_points_r)[1]):
            if sonar_points_r[2,i] == 1:
                # rotate by 90 degree 
                self.axis_sonar.plot(-sonar_points_r[1,i],sonar_points_r[0,i],'bx')

        self.axis_sonar.set_xlim([-self.robot.sonar.range-1,self.robot.sonar.range+1])
        self.axis_sonar.set_ylim([-1,self.robot.sonar.range+1])
        self.axis_sonar.set_aspect('equal')
        self.axis_sonar.set_title('Sonar measurement')

        # plot robot velocity in the robot frame (rotate x-axis by 90 degree (upward) in the plot)
        h1 = self.axis_dvl.arrow(0.0,0.0,0.0,1.0, \
                       color='k', \
                       width = 0.01, \
                       head_width = 0.06, \
                       head_length = 0.1, \
                       length_includes_head=True, \
                       label='steer direction')
        # rotate by 90 degree
        h2 = self.axis_dvl.arrow(0.0,0.0,-abs_velocity_r[1],abs_velocity_r[0], \
                       color='r',width=0.01, head_width = 0.06, \
                       head_length = 0.1, length_includes_head=True, \
                       label='velocity wrt seafloor')
        x_range = np.max([2,np.abs(abs_velocity_r[1])])
        y_range = np.max([2,np.abs(abs_velocity_r[0])])
        self.axis_dvl.set_xlim([-x_range,x_range])
        self.axis_dvl.set_ylim([-1,y_range])
        self.axis_dvl.set_aspect('equal')
        self.axis_dvl.legend(handles=[h1,h2])
        self.axis_dvl.set_title('DVL measurement')

    def one_step(self,action):
        current_velocity = self.get_velocity(self.robot.x, self.robot.y)
        self.robot.update_state(action,current_velocity)
        # print(self.robot.x, self.robot.y, self.robot.speed, self.robot.theta, \
        #       np.linalg.norm(current_velocity), np.linalg.norm(self.robot.velocity))

        self.plot_robot()
        self.plot_measurements()

    def visualize_control(self,action):
        # update robot state and make animation when executing the action    
        actions = []
        for _ in range(self.robot.N-1):
            actions.append(action)

        self.animation = mpl.animation.FuncAnimation(self.fig,self.one_step,actions, \
                                                interval=100,repeat=False)

        plt.show(block=False)
