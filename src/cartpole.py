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
M_11 = m_cart + m_pole # total mass
M_12 = m_pole * l * cos_theta # coupling term to link the cart's horizontal motion to the pole's angular motion
M_21 = m_pole * l * cos_theta 
M_22 = m_pole * l**2 # moment of inertia of the pole around its pivot point

# Nonlinear terms
C_1 = -m_pole * l * dtheta**2 * sin_theta # Centrifugal/Coriolis force term
C_2 = m_pole * g * l * sin_theta # gravity term pulling downward

det = M_11 * M_22 - M_12 * M_21
M_11_inv = M_22 / det
M_12_inv = -M_12 / det
M_21_inv = -M_21 / det
M_22_inv = M_11 / det

# Acceleration equations
x_ddot = M_11_inv * (F - C_1) + M_12_inv * -1 * C_2
theta_ddot = M_21_inv * (F - C_1) + M_22_inv * -1 * C_2

# Arbitrary-Body Algorithm function
aba_fn = casadi.Function("aba_fn", [cq, cq_dot, cu], [casadi.vertcat(x_ddot, theta_ddot)])

# Euler integrator
def euler_integrate(q, v, u):
    q_next = q + v * dt
    v_next = v + aba_fn(q, v, u) * dt
    return q_next, v_next

# optimization
opti = casadi.Opti()

Q = opti.variable(nq, N + 1)
V = opti.variable(nq_dot, N + 1)
U = opti.variable(1, N)

