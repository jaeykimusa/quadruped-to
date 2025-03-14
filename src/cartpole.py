import casadi
import numpy as np

# Variables
m_cart = 1.0 # mass of cart
m_pole = 0.5 # mass of pole
l = 1.0 # length of pole (to CoM)
g = 9.81 # m/s^2

# Optimization variables
t = 3.0 # time horizon
N = 100 # number of control intervals
dt = t / N


# Dynamics
nq = 2 # [cart position, pole angle]
nq_dot = 2 # [cart velocity, cart angular velocity]
cq = casadi.SX.sym("q", nq) # position matrix
cq_dot = casadi.SX.sym("v", nq_dot) # velocity matrix
cu = casadi.SX.sym("u", 1) # force on cart

x, theta = cq[0], cq[1] # variables from cart position matrix
dx, dtheta = cq_dot[0], cq_dot[1] # variables from cart velocity matrix
F = cu[0] # force

sin_theta = casadi.sin(theta)
cos_theta = casadi.cos(theta)


# Mass matrix
M11 = m_cart + m_pole
MM12 = m_pole * l * cos_theta
M21 = m_pole * l * cos_theta
M22 = m_pole * l**2


