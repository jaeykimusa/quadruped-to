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



import rerun as rr
from mpac_rerun.robot_logger import RobotLogger
rr.init("simple_robot_example", spawn=False)
robot_logger = RobotLogger.from_zoo("go2")

import time
current_time = time.time()
robot_logger.log_initial_state(logtime=current_time)


q = np.transpose(Q_sol)
from scipy.spatial.transform import Rotation as R

print(robot_logger.joint_names)

for i in range(q.shape[0]):
    q_i = q[i, :]
    r = R.from_euler("y", -q_i[2], degrees=False).as_quat()  # x, y, z w
    print(r)

    base_position = [q_i[0], 0, q_i[1]]
    base_orientation = r

    t3, t4, t1, t2 = q_i[3:]
    t1, t2, t3, t4 = q_i[3:]

    joint_positions = {
        "FL_hip_joint" : 0,
        "FL_thigh_joint" : -t1,
        "FL_calf_joint" : -t2,
        "FR_hip_joint" : 0,
        "FR_thigh_joint" : -t1,
        "FR_calf_joint" : -t2,
        "RL_hip_joint" : 0,
        "RL_thigh_joint" : -t3,
        "RL_calf_joint" : -t4,
        "RR_hip_joint" : 0,
        "RR_thigh_joint" : -t3,
        "RR_calf_joint" : -t4,
    }

    robot_logger.log_state(
        logtime=current_time,
        base_position=base_position,
        base_orientation=base_orientation,
        joint_positions=joint_positions
    )

    current_time += dt

rr.save("simple_robot.rrd")

exit()


# Simulation
sim = QuadrupedSimulator(robot, Q_sol, V_sol, U_sol, dt, N, user_prefs)
sim.simulate()
sim.export("jumpjumpjump_test.gif")


# visualizer.set_grf_factor(300) # scaling factor for large magnitude of grf

# if show_pitch_angle_over_time_plot:
#     x_time_array = np.linspace(0, dt * Q_sol.shape[1], Q_sol.shape[1])
#     y_pitch_angle = np.rad2deg(Q_sol[2, :])

#     plt.figure(figsize=(8, 4))
#     plt.plot(x_time_array, y_pitch_angle, label='Pitch Angle (rad)', color='blue')
#     plt.title("Pitch Angle Over Time")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Pitch Angle (deg)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # plt.savefig("location/title.png")


# # User preferences for vis
# show_trajectory = True          # was show_traj_line
# show_ground_plane = True        # was show_fixed_ground
# show_ground_reaction_forces = True  # was show_grf
# show_simulation_timer = True    # was show_timer
# vis_prefs = [show_trajectory, show_ground_plane, show_ground_reaction_forces, show_simulation_timer]


# # user preferences for data output
# export_grf_data = False         # was show_grf_output
# output_prefs = [export_grf_data]

# # user preferences for plot
# plot_pitch_angle = False
# plot_grf_over_time = False
# plot_joint_torque = False
# plot_prefs = [plot_pitch_angle, plot_grf_over_time, plot_joint_torque]

# # User preferences for downloads
# save_animated_gifs = False      # was save_gifs
# save_prefs = [save_animated_gifs]

# user_prefs = [vis_prefs, output_prefs, plot_prefs, save_prefs]
