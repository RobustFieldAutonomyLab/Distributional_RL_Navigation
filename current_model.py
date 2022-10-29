import numpy as np
import scipy
import matplotlib.pyplot as plt

class Core:

    def __init__(self, x:float, y:float, clockwise:bool, Gamma:float):

        self.x = x  # x coordinate of the vortex core  
        self.y = y  # y coordinate of the vortex core
        self.clockwise = clockwise # if the vortex direction is clockwise
        self.Gamma = Gamma  # circulation strength of the vortex core

class Map:

    def __init__(self, cores:list):
        
        # parameter initialization
        self.width = 100 # x coordinate dimension of the map
        self.height = 100 # y coordinate dimension of the map
        self.r = 0.5  # radius of vortex core
        self.v_rel_max = 0.5 # max allowable speed when two currents flowing towards each other
        self.p = 0.8 # max allowable relative speed at another vortex core
        self.cores = [] # vertex cores 

        for i,core in enumerate(cores):
            if self.check_core(core):
                self.cores.append(core)
            else:
                raise RuntimeError("core "+str(i)+" is not viable")

        centers = None
        for core in self.cores:
            if centers is None:
                centers = np.array([[core.x,core.y]])
            else:
                c = np.array([[core.x,core.y]])
                centers = np.vstack((centers,c))
        
        # KDTree storing vortex core center positions
        self.core_centers = scipy.spatial.KDTree(centers)

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

    def get_velocity(self,x:float, y:float):
        # find the closest vortex
        d, idx = self.core_centers.query(np.array([x,y]))
        v_base_radial = np.matrix([[self.cores[idx].x-x],[self.cores[idx].y-y]])
        v_base_radial /= d

        v_velocity = np.zeros((2,1))
        for i,core in enumerate(self.cores): 
            v_radial = np.matrix([[core.x-x],[core.y-y]])

            if i != idx:
                # if the vortex is in the outter area of the closest vortex, 
                # exclude it from velocity computation 
                project = np.transpose(v_radial)*v_base_radial
                if project[0,0] > d:
                    continue

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
        ax.quiver(pos_x, pos_y, arrow_x, arrow_y)
        
        plt.show()
