import numpy as np
import env

class Sonar:

    def __init__(self):
        # 2D model with detection area as a sector
        self.range = 10 # meter
        self.angle = 2 * np.pi / 3
        self.num_beams = 11 # number of beams (asssume each is a line)
        self.phi = self.angle / (self.num_beams-1) # interval angle between two beams
        self.beam_angles = [] # relative angles to the center
        self.reflections = [] # reflection points and indicators

        angle = -self.angle/2
        for i in range(self.num_beams):
            angle += i * self.phi
            self.beam_angles.append(angle)

class Robot:

    def __init__(self):
        self.dt = 0.1 # discretized time step (second)
        self.N = 10 # number of time step per action
        self.sonar = Sonar
        self.length = 1.0 
        self.width = 0.5
        self.x = None # x coordinate
        self.y = None # y coordinate
        self.theta = None # steering heading angle
        self.speed = None # steering forward speed
        self.velocity = None # velocity wrt sea floor

    def set_state(self,x,y,theta=0.0,speed=0.0,current_velocity=np.zeros(2)):
        self.x = x
        self.y = y
        self.theta = theta 
        self.speed = speed
        self.update_velocity(current_velocity) 

    def get_robot_transform(self):
        # compute transformation from world frame to robot frame
        R_wr = np.matrix([[np.cos(self.theta) -np.sin(self.theta)],[np.sin(self.theta),np.cos(self.theta)]])
        t_wr = np.matrix([[self.x],[self.y]])
        return R_wr, t_wr

    def get_steer_velocity(self):
        return self.speed * np.array(np.cos(self.theta), np.sin(self.theta))

    def update_velocity(self,current_velocity=np.zeros(2)):
        steer_velocity = self.get_steer_velocity()
        self.velocity = steer_velocity + current_velocity

    def update_state(self,action,current_velocity):
        # update robot position in one time step
        self.update_velocity(current_velocity)
        dis = self.velocity * self.dt
        self.x += dis[0]
        self.y += dis[1]
        
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

    def sonar_reflection(self,obstacles):
        
        # if a beam does not have reflection, set the reflection point distance
        # twice as long as the beam range, and set indicator to 0
        self.sonar.reflections.clear()

        for rel_a in self.sonar.beam_angles:
            angle = self.theta + rel_a

            # initialize the beam reflection as null
            if np.abs(angle - np.pi/2) < 1e-03 or \
                np.abs(angle + 3*np.pi/2) < 1e-03:
                d = 2.0 if np.abs(angle - np.pi/2) < 1e-03 else -2.0
                x = self.x
                y = self.y + d*self.sonar.range
            else:
                d = 2.0
                x = self.x + d * self.sonar.range * np.cos(angle)
                y = self.y + d * self.sonar.range * np.sin(angle)

            self.sonar.reflections.append([x,y,0])
            
            # compute the beam reflection given obstcale  
            for obs in obstacles:
                if self.sonar.reflections[-1][-1] != 0:
                    break
                if np.abs(angle - np.pi/2) < 1e-03 or \
                    np.abs(angle + 3*np.pi/2) < 1e-03:
                    # vertical line
                    M = obs.r*obs.r - (self.x-obs.x)*(self.x-obs.x)
                    
                    if M < 0.0:
                        # no real solution
                        continue
                    
                    x1 = self.x
                    x2 = self.x
                    y1 = obs.y - np.sqrt(M)
                    y2 = obs.y + np.sqrt(M)
                else:
                    K = np.tan(angle)
                    a = 1 + K*K
                    b = 2*K*(self.y-K*self.x-obs.y)-2*obs.x
                    c = obs.x*obs.x + \
                        (self.y-K*self.x-obs.y)*(self.y-K*self.x-obs.y) - \
                        obs.r*obs.r
                    delta = b*b - 4*a*c
                    
                    if delta < 0.0:
                        # no real solution
                        continue
                    
                    x1 = (-b-np.sqrt(delta))/(2*a)
                    x2 = (-b+np.sqrt(delta))/(2*a)
                    y1 = self.y+K*(x1-self.x)
                    y2 = self.y+K*(x2-self.x)

                dis1 = np.sqrt((x1-self.x)*(x1-self.x)+(y1-self.y)*((y1-self.y)))
                dis2 = np.sqrt((x2-self.x)*(x2-self.x)+(y2-self.y)*((y2-self.y)))
                if np.min(dis1,dis2) > self.sonar.range:
                    # beyond detection range
                    continue
                if dis1 < dis2:
                    self.sonar.reflections[-1] = [x1,y1,1]
                else:
                    self.sonar.reflections[-1] = [x2,y2,1]


                     



