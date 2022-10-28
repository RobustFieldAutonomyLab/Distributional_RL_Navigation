from tokenize import Double


class Core:

    def __init__(self, x:float, y:float, clockwise:bool, w:float):

        self.x = x  # x coordinate of the vortex core  
        self.y = y  # y coordinate of the vortex core
        self.clockwise = clockwise # if the vortex direction is clockwise
        self.w = w  # angular velocity of the vortex core

class Rankine:

    def __init__(self, cores):
        
        # parameter initialization
        self.r = 1  # radius of vortex core
        self.k = 1  # velocity decay rate

        for core in cores:
            

    def visualization():
        pass