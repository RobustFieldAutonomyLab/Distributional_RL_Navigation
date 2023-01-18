import numpy as np

class BA_agent:

    def __init__(self,a,w):
        # self.state = "forward"
        # self.follow_dist = 5.0 # distance to the wall when following

        self.a = a # available linear acceleration (action 1)
        self.w = w # available angular velocity (action 2)

    def act(self, observation):
        velocity = observation[:2]
        if np.linalg.norm(velocity) < 1e-03:
            velocity = np.ones(2)

        goal = observation[2:4]
        sonar_points = observation[4:]

        w_idx = self.wall_follow_action(velocity,sonar_points)

        if w_idx is None:
            # No obstacles detected, move towards the goal
            G_angle = np.arctan2(goal[1],goal[0])
            V_angle = np.arctan2(velocity[1],velocity[0])
            diff_angle = wrap_to_pi(G_angle - V_angle)
            w_idx = np.argmin(np.abs(self.w-diff_angle))
        
        a_idx = 2

        return a_idx * len(self.w) + w_idx
            

    def wall_follow_action(self, velocity, sonar_points):
        # compute the linear regression wall function, 
        # return the action to follow the wall
        
        A = None
        b = None
        for i in range(0,len(sonar_points),2):
            x = sonar_points[i]
            y = sonar_points[i+1]

            if x == 0 and y == 0:
                continue
            
            x_array = np.array([x,1])
            y_array = np.array([y])

            if A is None:
                A = np.array([x_array])
                b = np.array([y_array])
            else:
                A = np.vstack((A,x_array))
                b = np.vstack((b,y_array))

        if A is None:
            return None
        elif np.shape(A)[0] == 1:
            # distance to wall
            d = np.linalg.norm(A)
            
            # wall tangent vector
            R = np.matrix([[0.,-1.],[1.,0.]])
            dir = R * np.reshape(A[0],(2,1))
            dir = np.array([dir[0,0],dir[1,0]])            
        elif np.shape(A)[0] == 2:
            # vector to a point on wall
            v_1 = A[0]
            
            # wall tangent vector
            dir = A[1]-A[0]

            # distance to wall
            cross = np.abs(np.cross(v_1,dir))
            d = cross / np.linalg.norm(dir)
        else:
            A = np.matrix(A)
            b = np.matrix(b)
            
            At_A = np.transpose(A)*A
            
            # check if the wall is vertical
            _,s,_ = np.linalg.svd(At_A)
            if s[1] < 1e-03 * s[0]:
                # wall tangent vector
                dir = np.array([0.0,1.0])
                
                # distance to wall
                d = np.abs(np.mean(A[:,0]))
            else:
                # linear regression (A'*A)^(-1) * A' * b
                res = np.linalg.inv(At_A)*np.transpose(A)*b
                
                # vector to a point on wall
                v_1 = np.array([1.0,np.sum(res)])

                # wall tangent vector
                dir = np.array([1.0,res[0,0]])

                # distance to wall
                cross = np.abs(np.cross(v_1,dir))
                d = cross / np.linalg.norm(dir)

        # reverse the tangent vector if diff angle is greater than 90 degree
        if np.dot(dir,velocity) < 0:
            dir *= -1.0

        W_angle = np.arctan2(dir[1],dir[0])
        V_angle = np.arctan2(velocity[1],velocity[0])
        diff_angle = wrap_to_pi(W_angle - V_angle)
        w_idx = np.argmin(np.abs(self.w-diff_angle))

        return w_idx

def wrap_to_pi(angle):
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle >= np.pi:
        angle -= 2 * np.pi
    return angle

            


