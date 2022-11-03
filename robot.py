import numpy as np

class Sonar:

    def __init__(self):
        # 2D model with detection area as a sector
        self.range = 10 # meter
        self.angle = 2 * np.pi / 3
        self.num_beams = 11 # number of beams (asssume each is a line)
        self.phi = self.angle / (self.num_beams-1) # interval angle between two beams
        self.beam_angles = [] # relative angles to the center

        angle = -self.angle/2
        for i in range(self.num_beams):
            angle += i * self.phi
            self.beam_angles.append(angle)

class Robot:

    def __init__(self):
        self.dt = 0.1 # discretized time step (second)
        self.N = 10 # number of time step per action
        self.sonar = Sonar

    def set_state(self,pos_x,pos_y,theta=0.0,speed=0.0):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.theta = theta # heading angle
        self.speed = speed # forward speed

    def update_state(self,action,current_velocity):
        # update robot position in one time step
        steer_velocity = self.speed * np.array(np.cos(self.theta), np.sin(self.theta))
        total_velocity = steer_velocity + current_velocity
        dis = total_velocity * self.dt
        self.pos_x += dis[0]
        self.pos_y += dis[1]
        
        # update robot heading angle and forward speed in one time step
        a = action[0]
        w = action[1]
        self.speed += a * self.dt
        self.theta += w * self.dt

        # warp theta to [0,2*pi)
        while self.theta < 0.0:
            self.theta += 2 * np.pi
        while self.theta >= 2 * np.pi:
            self.theta -= 2 * np.pi

    def sonar_reflection(self,obs_x,obs_y,obs_r):
        pass

