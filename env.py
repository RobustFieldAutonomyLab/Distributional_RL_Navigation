import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

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

class Env:

    def __init__(self, seed:int=0):
        
        # parameter initialization
        self.rd = np.random.RandomState(seed) # PRNG 
        self.width = 50 # x coordinate dimension of the map
        self.height = 50 # y coordinate dimension of the map
        self.r = 0.5  # radius of vortex core
        self.v_rel_max = 0.5 # max allowable speed when two currents flowing towards each other
        self.p = 0.8 # max allowable relative speed at another vortex core
        self.v_range = [5,10] # speed range of the vortex (at the edge of core)
        self.obs_r_range = [1,5] # radius range of the obstacle
        self.cores = [] # vertex cores
        self.obstacles = [] # cylinder obstacles 

        self.reset()

    def reset(self, num_cores:int = 5, num_obs:int = 5):
        # reset the environment
        
        self.cores.clear()
        self.obstacles.clear()

        # generate vortex with random position, spinning direction and strength
        while True:
            center = self.rd.uniform(low = np.zeros(2), high = np.array([self.width,self.height]))
            direction = self.rd.binomial(1,0.5)
            v_edge = self.rd.uniform(low = self.v_range[0], high = self.v_range[1])
            Gamma = 2 * np.pi * self.r * v_edge
            core = Core(center[0],center[1],direction,Gamma)
            if self.check_core(core):
                self.cores.append(core)
                num_cores -= 1
            if num_cores == 0:
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
        while True:
            center = self.rd.uniform(low = np.zeros(2), high = np.array([self.width,self.height]))
            r = self.rd.uniform(low = self.obs_r_range[0], high = self.obs_r_range[1])
            obs = Obstacle(center[0],center[1],r)
            if self.check_obstacle(obs):
                self.obstacles.append(obs)
                num_obs -= 1
            if num_obs == 0:
                break

    def step(self, action):
        # execute action and update the environment
        pass

    def get_obs(self):
        # provide observation for the agent
        pass

    def get_reward(self):
        # return reward
        pass

    def check_core(self,core_j):

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
        
        return v_velocity

    def compute_speed(self, Gamma:float, d:float):
        if d <= self.r:
            return Gamma / (2*np.pi*self.r*self.r) * d
        else:
            return Gamma / (2*np.pi*d)          

    def visualization(self):
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
                arrow_x.append(v[0,0])
                arrow_y.append(v[1,0])
        
        fig, ax = plt.subplots()
        
        # plot current velocity
        ax.quiver(pos_x, pos_y, arrow_x, arrow_y)

        # plot obstacles
        for obs in self.obstacles:
            ax.add_patch(mpl.patches.Circle((obs.x,obs.y),radius=obs.r))
        
        ax.set_aspect('equal')

        plt.show()
