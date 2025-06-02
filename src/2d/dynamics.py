# dynamics.py


import numpy as np

# from robot import m, g, I_z
from kinematics import forward_kinematics
from robot import robot


def euler_integrate(q, v, u, dt):
    # Variables
    x, y, phi = q[0], q[1], q[2] # positions
    vx, vy, omega = v[0], v[1], v[2] # velocities
    theta1, theta2, theta3, theta4 = q[3], q[4], q[5], q[6] # joint angles
    dtheta1, dtheta2, dtheta3, dtheta4 = u[4], u[5], u[6], u[7] # angular velocities for joints
    GRF1_x, GRF1_y, GRF2_x, GRF2_y = u[0], u[1], u[2], u[3] # ground reaction forces

    # Integrate for positions
    x_next = x + vx * dt
    y_next = y + vy * dt
    phi_next = phi + omega * dt

    # Integrate for joint angles
    theta1_next = theta1 + dtheta1 * dt
    theta2_next = theta2 + dtheta2 * dt
    theta3_next = theta3 + dtheta3 * dt
    theta4_next = theta4 + dtheta4 * dt

    # Translational acceleration using newton's law
    ax = (GRF1_x + GRF2_x) / robot.get_m()
    ay = (GRF1_y + GRF2_y) / robot.get_m() - robot.g

    fk = forward_kinematics(q)

    # Compute position of the feet relative to the CoM
    rfront_x = fk.front_ankle_x - fk[0]
    rfront_y = fk.front_ankle_y - fk[1]
    rrear_x = fk.rear_ankle_x - fk[0]
    rrear_y = fk.rear_ankle_y - fk[1]

    # net torque from the point force, GRF1 and FRF2.
    torque = (-GRF1_x * rfront_y + GRF1_y * rfront_x) + (-GRF2_x * rrear_y + GRF2_y * rrear_x)
    # angular acc
    alpha = torque / robot.I_z

    vx_next = vx + ax * dt
    vy_next = vy + ay * dt
    omega_next = omega + alpha * dt # angular acceleration of the body caused by external torques from the foot-ground forces (?)

    q_next = np.array([x_next, y_next, phi_next, theta1_next, theta2_next, theta3_next, theta4_next])
    v_next = np.array([vx_next, vy_next, omega_next])

    return q_next, v_next


def rk4_integrate(q, v, u, dt):
    # Variables
    x, y, phi = q[0], q[1], q[2] # positions
    vx, vy, omega = v[0], v[1], v[2] # velocities
    theta1, theta2, theta3, theta4 = q[3], q[4], q[5], q[6] # joint angles
    dtheta1, dtheta2, dtheta3, dtheta4 = u[4], u[5], u[6], u[7] # angular velocities for joints
    GRF1_x, GRF1_y, GRF2_x, GRF2_y = u[0], u[1], u[2], u[3] # ground reaction forces

    # Integrate for positions
    x_next = x + vx * dt
    y_next = y + vy * dt
    phi_next = phi + omega * dt

    # Integrate for joint angles
    theta1_next = theta1 + dtheta1 * dt
    theta2_next = theta2 + dtheta2 * dt
    theta3_next = theta3 + dtheta3 * dt
    theta4_next = theta4 + dtheta4 * dt

    # Translational acceleration using newton's law
    ax = (GRF1_x + GRF2_x) / robot.get_m()
    ay = (GRF1_y + GRF2_y) / robot.get_m() - robot.g

    fk = forward_kinematics(q)

    # Compute position of the feet relative to the CoM
    rfront_x = fk.front_ankle_x - fk[0]
    rfront_y = fk.front_ankle_y - fk[1]
    rrear_x = fk.rear_ankle_x - fk[0]
    rrear_y = fk.rear_ankle_y - fk[1]

    # net torque from the point force, GRF1 and FRF2.
    torque = (-GRF1_x * rfront_y + GRF1_y * rfront_x) + (-GRF2_x * rrear_y + GRF2_y * rrear_x)
    # angular acc
    alpha = torque / robot.I_z

    vx_next = vx + ax * dt
    vy_next = vy + ay * dt
    omega_next = omega + alpha * dt # angular acceleration of the body caused by external torques from the foot-ground forces (?)

    q_next = np.array([x_next, y_next, phi_next, theta1_next, theta2_next, theta3_next, theta4_next])
    v_next = np.array([vx_next, vy_next, omega_next])

    return q_next, v_next

def runge_kutta_4():
    