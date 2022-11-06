import numpy as np
import env

if __name__ == "__main__":
    # core_1 = current_model.Core(25,75,True,10)
    # core_2 = current_model.Core(75,75,False,20)
    # core_3 = current_model.Core(50,50,True,40)
    # core_4 = current_model.Core(25,25,False,15)
    # core_5 = current_model.Core(75,25,True,5)

    # map = current_model.Map([core_1,core_2,core_3,core_4,core_5])
    map = env.Env()
    # current_v = map.get_velocity(30.0,20.0)
    # map.robot.set_state(30.0,20.0,np.pi/6,current_velocity=current_v)
    map.init_visualize()
    map.visualize_control([15.0,0.785])