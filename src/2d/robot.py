# robot.py


import numpy as np


# Physical constants
# Physical parameters
m = 32.7  # kg boston dynamic's spot
L = 1.1  # body length (m)
h = 0.1 # body heigh (m) - ideally 0.2
I_z = 1 / 12 * m * (L**2 + h**2)  # moment of inertia for rectangular body around the z-axis 
leg_length = 0.4  # m - ideally 0.3-0.35


# friction
g = 9.81
mu = 0.8 # default coefficient of friction


# dimensions
n_q = 7 # 7 dof
n_v = 3 # spatial velocity
n_u = 8 # 4 ground reaction forces + 4 joint velocities