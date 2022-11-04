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
        self.theta = theta # steering heading angle
        self.speed = speed # steering forward speed

    def get_robot_transform(self):
        # compute transformation from world frame to robot frame
        R_wr = np.matrix([[np.cos(self.theta) -np.sin(self.theta)],[np.sin(self.theta),np.cos(self.theta)]])
        t_wr = np.matrix([[self.pos_x],[self.pos_y]])
        return R_wr, t_wr

    def get_steer_velocity(self):
        return self.speed * np.array(np.cos(self.theta), np.sin(self.theta))

    def update_state(self,action,current_velocity):
        # update robot position in one time step
        steer_velocity = self.get_steer_velocity()
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
        
        res = []

        for rel_a in self.sonar.beam_angles:
            angle = self.theta + rel_a
            
            if np.abs(angle - np.pi/2) < 1e-03 or \
               np.abs(angle + 3*np.pi/2) < 1e-03:
                # vertical line
                M = obs_r*obs_r - (self.pos_x-obs_x)*(self.pos_x-obs_x)
                if M < 0.0:
                    # no real solution
                    continue
                x1 = self.pos_x
                x2 = self.pos_x
                y1 = obs_y - np.sqrt(M)
                y2 = obs_y + np.sqrt(M)
            else:
                K = np.tan(angle)
                a = 1 + K*K
                b = 2*K*(self.pos_y-K*self.pos_x-obs_y)-2*obs_x
                c = obs_x*obs_x + \
                    (self.pos_y-K*self.pos_x-obs_y)*(self.pos_y-K*self.pos_x-obs_y) - \
                    obs_r*obs_r
                delta = b*b - 4*a*c
                if delta < 0.0:
                    # no real solution
                    continue
                x1 = (-b-np.sqrt(delta))/(2*a)
                x2 = (-b+np.sqrt(delta))/(2*a)
                y1 = self.pos_y+K*(x1-self.pos_x)
                y2 = self.pos_y+K*(x2-self.pos_x)

            dis1 = np.sqrt((x1-self.pos_x)*(x1-self.pos_x)+(y1-self.pos_y)*((y1-self.pos_y)))
            dis2 = np.sqrt((x2-self.pos_x)*(x2-self.pos_x)+(y2-self.pos_y)*((y2-self.pos_y)))
            if np.min(dis1,dis2) > self.sonar.range:
                # beyond detection range
                continue
            if dis1 < dis2:
                res.append([x1,y1])
            else:
                res.append([x2,y2])

        # fixed 

        return res


                     



