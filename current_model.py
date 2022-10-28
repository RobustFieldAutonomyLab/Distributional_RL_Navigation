import numpy as np
import scipy
import matplotlib.pyplot as plt

class Core:

    def __init__(self, x:float, y:float, clockwise:bool, w:float):

        self.x = x  # x coordinate of the vortex core  
        self.y = y  # y coordinate of the vortex core
        self.clockwise = clockwise # if the vortex direction is clockwise
        self.w = w  # angular velocity of the vortex core

class Map:

    def __init__(self, cores:list):
        
        # parameter initialization
        self.width = 100 # x coordinate dimension of the map
        self.height = 100 # y coordinate dimension of the map
        self.r = 0.5  # radius of vortex core
        self.k = 1  # velocity decay rate
        self.cores = cores # vertex cores 

        centers = None
        for core in self.cores:
            if centers is None:
                centers = np.array([[core.x,core.y]])
            else:
                c = np.array([[core.x,core.y]])
                centers = np.vstack((centers,c))
        
        # KDTree storing vortex core center positions
        self.core_centers = scipy.spatial.KDTree(centers)

    def get_velocity(self,x:float, y:float):
        nn = 1
        dis, idx = self.core_centers.query(np.array([[x,y]]),k=nn)
        v_velocity = np.zeros((2,1))
        for i in range(nn):
            core = self.cores[idx[i]]
            v_radial = np.matrix([[core.x-x],[core.y-y]])
            v_radial /= np.linalg.norm(v_radial)
            if core.clockwise:
                rotation = np.matrix([[0., -1.],[1., 0]])
            else:
                rotation = np.matrix([[0., 1.],[-1., 0]])
            v_tangent = rotation * v_radial
            speed = self.compute_speed(core.w,dis[i])
            v_velocity += v_tangent * speed
        
        return v_velocity

    def compute_speed(self, w:float, d:float):
        if d <= self.r:
            return w * d
        else:
            return w * self.r - self.k / self.r + self.k / d            

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
