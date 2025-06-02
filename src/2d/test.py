# simulation.py


from robot import Robot, robot
from dynamics import euler_integrate
from control import *

from vis import QuadrupedSimulator
from kinematics import forward_kinematics
import casadi as ca
import numpy as np



# initial state
q_initial = np.array([0.0, 0.6, 0, -1, 1.4, -1, +1.4])
v_initial = np.array([0, 0, 0])

# final state
q_final = np.array([2.5, 0.6, 0, -1, 1.4, -1, +1.4])
v_final = np.array([0, 0, 0])

# user preferences
user_prefs = {
    "visualization": {
        "show_trajectory": True,
        "show_fixed_ground_plane": True,
        "show_grf": True,
        "show_simulation_timer": True
    },
    "data_output": {
        "export_grf_data": False
    },
    "plot_settings": {
        "plot_pitch_angle": False,
        "plot_grf_over_time": False,
        "plot_joint_torque": False
    },
    "export_settings": {
        "save_animated_gifs": True
    }
}

dt = 0.02
N = 100


# cost function
cost_func = 0


# Cost function weights of corresponding elements
JOINT_ANGLE_PENALTY_WEIGHT = 1
JOINT_VELOCITY_PENALTY_WEIGHT = 1
GRF_PENALTY_WEIGHT = 0.001
BODY_ORIENTATION_PENALTY_WEIGHT = 0.001


q = ca.SX.sym("q", robot.n_q)
v = ca.SX.sym("v", robot.n_v)
u = ca.SX.sym("u", robot.n_u)


q_next, v_next = euler_integrate(q, v, u, dt)

get_next_state = ca.Function("get_next_state", [q, v, u], [q_next, v_next])

opti = ca.Opti()


# Simulated time in seconds
time_array = np.linspace(0, N * dt, N)  

Q = opti.variable(robot.n_q, N)
V = opti.variable(robot.n_v, N)
U = opti.variable(robot.n_u, N - 1)


opti.subject_to(Q[:, 0] == q_initial)
opti.subject_to(V[:, 0] == v_initial)

opti.subject_to(Q[:, -1] == q_final)
opti.subject_to(V[:, -1] == v_final)


# All feet contact phase
ALL_FEET_CONTACT_START = 0
ALL_FEET_CONTACT_END = int(0.3 * N)

# Rear feet contact phase
REAR_FEET_CONTACT_START = ALL_FEET_CONTACT_END
REAR_FEET_CONTACT_END = int(0.4 * N)

# Flight phase
FLIGHT_START = REAR_FEET_CONTACT_END
FLIGHT_END = int(0.75 * N)

# Landing phase
LANDING_START = FLIGHT_END 
LANDING_END = N

for k in range(N - 1):
    
    add_dynamics_constraint(opti, Q, V, U, k, get_next_state)

    # Phase 1 - All feet contact
    if ALL_FEET_CONTACT_START <= k < ALL_FEET_CONTACT_END:
        add_friction_cone_constraint(opti, U, k, robot.mu, "front")
        add_contact_constraint(opti, Q, k, q_initial, "front")

        add_friction_cone_constraint(opti, U, k, robot.mu, "rear")
        add_contact_constraint(opti, Q, k, q_initial, "rear")

    # Phase 2 - Rear feet only
    if REAR_FEET_CONTACT_START <= k < REAR_FEET_CONTACT_END:
        add_zero_force_constraint(opti, U, k, "front")

        add_friction_cone_constraint(opti, U, k, robot.mu, "rear")
        add_contact_constraint(opti, Q, k, q_initial, "rear")

    # Phase 3 - Flight
    if FLIGHT_START <= k < FLIGHT_END:
        add_zero_force_constraint(opti, U, k, "front")

        add_zero_force_constraint(opti, U, k, "rear")

    # Phase 4 - Landing
    if LANDING_START <= k < LANDING_END:
        add_friction_cone_constraint(opti, U, k, robot.mu, "front")
        add_contact_constraint(opti, Q, k, q_final, "front")

        add_friction_cone_constraint(opti, U, k, robot.mu, "rear")
        add_contact_constraint(opti, Q, k, q_final, "rear")


def is_contact(n):
    return n < int(0.35 * N) or n > int(0.65 * N)


for k in range(N - 1):
    q_angles = Q[3:, k]
    q_angles_default = q_initial[3:]
    q_angles_diff = q_angles - q_angles_default

    lower_bounds = [-np.pi / 2] * 4
    uppder_bounds = [np.pi / 2] * 4
    apply_joint_limits(opti, q_angles_diff, lower_bounds, uppder_bounds)

    cost_func += JOINT_ANGLE_PENALTY_WEIGHT * ca.dot(q_angles_diff, q_angles_diff)
    uext = U[:4, k]
    cost_func += GRF_PENALTY_WEIGHT * ca.dot(uext, uext)
    uext = U[4:, k]
    cost_func += JOINT_VELOCITY_PENALTY_WEIGHT  * ca.dot(uext, uext)

    phi = Q[2, k]

    omega = k / N

    cost_func += BODY_ORIENTATION_PENALTY_WEIGHT * (phi - (q_initial[2] + omega * (q_final[2] - q_initial[2]))) ** 2

opti.minimize(cost_func)

# intial guess
initial_guess(opti, Q, V, U, q_initial, q_final, v_initial, v_final, robot.get_m(), robot.g, N, is_contact)

# solve via ipopt
opti.solver("ipopt")

# full optimized trajectory
Q_sol = opti.solve().value(Q)
V_sol = opti.solve().value(V)
U_sol = opti.solve().value(U)

# print("Shape:", Q_sol.shape)
# print("Shape:", V_sol.shape)
# print("Shape:", U_sol.shape)


# Outputs for user preferences
# user_output_preferences(SHOW_GRF_OUTPUT)


# if SHOW_GRF_OUTPUT:
#         print("GRF over time:")

#         for i in range (N - 1):
#             t = time_array[i]
#             F1x = U_sol[0, i]
#             F1y = U_sol[1, i]
#             F2x = U_sol[2, i]
#             F2y = U_sol[3, i]

#             print(f"Time: {t:0.3f}s - F1x: {F1x: .3f}N, F1y: {F1y: .3f}N, F2x: {F2x: .3f}N, F2y: {F2y: .3f}N")



# Simulation
sim = QuadrupedSimulator(robot, Q_sol, V_sol, U_sol, dt, N, user_prefs)
sim.simulate()
sim.export("jumpjumpjump_test.gif")