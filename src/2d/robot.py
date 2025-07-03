# robot.py


import numpy as np


ROBOT_MASS = 15.210 # # kg
ROBOT_BODY_LENGTH = 2 * 0.1934
ROBOT_BODY_HEIGHT = 0.05
ROBOT_LEG_LENGTH = 0.213


class Robot:
    def __init__(self, m, L, h, l_length):
        
        # Physical parameters
        self.m = m # kg boston dynamic's spot
        self.L = L # body length (m)
        self.h = h # body heigh (m) - ideally 0.2
        self.I_z = 1 / 12 * m * (L**2 + h**2)  # moment of inertia for rectangular body around the z-axis 
        self.leg_length = l_length # m - ideally 0.3-0.35


        # friction
        self.g = 9.81
        self.mu = 0.8 # default coSfficient of friction


        # dimensions
        self.n_q = 7 # 7 dof
        self.n_v = 3 # spatial velocity
        self.n_u = 8 # control input vector: 4 ground reaction forces + 4 joint torques (velocities)
    
    def get_m(self):
        return self.m
    
    def get_leg_length(self):
        return self.leg_length
    
    def get_L(self):
        return self.L

robot = Robot(ROBOT_MASS, ROBOT_BODY_LENGTH, ROBOT_BODY_HEIGHT, ROBOT_LEG_LENGTH)