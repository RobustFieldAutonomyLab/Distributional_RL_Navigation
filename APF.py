import numpy as np
import copy

class APF_agent:

    def __init__(self, a, w):
        self.k_att = 50.0 # attractive force constant
        self.k_rep = 500.0 # repulsive force constant
        self.m = 500 # robot weight (kg)
        self.d0 = 10.0 # obstacle distance threshold (m)
        self.n = 2 # power constant of repulsive force
        self.min_vel = 1.0 # if velocity is lower than the threshold, mandate acceleration 

        self.a = a # available linear acceleration (action 1)
        self.w = w # available angular velocity (action 2)

    def act(self, observation):
        velocity = observation[:2]
        goal = observation[2:4]
        sonar_points = observation[4:]

        # compute attractive force
        F_att = self.k_att * goal

        # compute total repulsive force from sonar reflections
        F_rep = np.zeros(2)
        d_goal = np.linalg.norm(goal)
        for i in range(0,len(sonar_points),2):
            x = sonar_points[i]
            y = sonar_points[i+1]

            if x == 0 and y == 0:
                continue
        
            d_obs = np.linalg.norm(sonar_points[i:i+2])

            # repulsive force component to move away from the obstacle 
            mag_1 = self.k_rep * ((1/d_obs)-(1/self.d0)) * (d_goal ** self.n)/(d_obs ** 2)
            dir_1 = -1.0 * sonar_points[i:i+2] / d_obs
            F_rep_1 = mag_1 * dir_1

            # repulsive force component to move towards the goal
            mag_2 = (self.n / 2) * self.k_rep * (((1/d_obs)-(1/self.d0))**2) * (d_goal ** (self.n-1))
            dir_2 = -1.0 * goal / d_goal
            F_rep_2 = mag_2 * dir_2

            F_rep += (F_rep_1 + F_rep_2)

        # select angular velocity action 
        F_total = F_att + F_rep
        V_angle = 0.0
        if np.linalg.norm(velocity) > 1e-03:
            V_angle = np.arctan2(velocity[1],velocity[0])
        F_angle = np.arctan2(F_total[1],F_total[0])

        diff_angle = F_angle - V_angle
        while diff_angle < -np.pi:
            diff_angle += 2 * np.pi
        while diff_angle >= np.pi:
            diff_angle -= 2 * np.pi

        w_idx = np.argmin(np.abs(self.w-diff_angle))
        
        # select linear acceleration action
        a_total = F_total / self.m
        V_dir = np.array([1.0,0.0])
        if np.linalg.norm(velocity) > 1e-03:
            V_dir = velocity / np.linalg.norm(velocity)
        a_proj = np.dot(a_total,V_dir)
 
        a = copy.deepcopy(self.a)
        if np.linalg.norm(velocity) < self.min_vel:
            # if the velocity is small, mandate acceleration
            a[a<=0.0] = -np.inf
        a_diff = a-a_proj
        a_idx = np.argmin(np.abs(a_diff))

        return a_idx * len(self.w) + w_idx

        
        
