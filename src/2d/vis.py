# visualization.py


from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from kinematics import forward_kinematics
from robot import Robot, robot


class QuadrupedSimulator:
    def __init__(self, robot, figax=None):
        # self.L = L
        # self.leg_length = leg_length
        # self.h = h

        self.robot = robot

        # Initialize the plot and axis
        if figax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig, self.ax = figax
        
        # Set plot appearance such size and orientn.
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-2.0, 4.5)
        self.ax.set_ylim(-1.0, 4.0)

        # Rectangle for body
        self.body_patch = patches.Polygon([[0, 0], [0, 0], [0, 0], [0, 0]], closed=True, color="#5FB257", alpha=0.5)
        self.ax.add_patch(self.body_patch)

        # CoM marker
        (self.com_plot,) = self.ax.plot([], [], "ko", markersize=3, label="CoM")

        # Joint markers
        (self.shoulder_plots,) = self.ax.plot([], [], "o", markersize=3, markerfacecolor='white', markeredgecolor='blue')
        (self.knee_plots,) = self.ax.plot([], [], "o", markersize=3, markerfacecolor='white', markeredgecolor='blue')
        (self.ankle_plots,) = self.ax.plot([], [], "o", markersize=3, markerfacecolor='white', markeredgecolor='blue')

        # Leg lines
        (self.link_lines_front,) = self.ax.plot([], [], "k-", linewidth=2)
        (self.link_lines_rear,) = self.ax.plot([], [], "k-", linewidth=2)

        # # GRFs
        # if grf:
        #     (self.force_lines_front,) = self.ax.plot([], [], "r-", linewidth=1)
        #     (self.force_lines_rear,) = self.ax.plot([], [], "r-", linewidth=1)

        # # Trajectory line
        # if trajectory_line:
        #     (self.q_trajectory,) = self.ax.plot([], [], "k--", lw=1, alpha=0.5)

        # Draw ground line
        self.ax.plot([-3.0, 5.0], [0, 0], "k-", linewidth=1)

        # # Timer
        # if timer:
        #     self.timer_text = self.ax.text(
        #         0.5, 1.02,                  # x and y in axis coordinates (top-center, a bit above)
        #         "Time: 0.00 s",             # Initial text
        #         transform=self.ax.transAxes,
        #         ha="center", va="bottom", fontsize=10, color="black", fontweight="bold"
        #     )

        # if fixed_ground: 
        #     x_start = -3.0
        #     x_end = 5.0
        #     spacing = 0.2
        #     length = 0.2
        #     x_vals = np.arange(x_start, x_end, spacing)
        #     for x in x_vals:
        #         self.ax.plot([x + length, x], [-0.01, -0.05 - length], "k-", linewidth=1)

    def set_data(self, q, u=None, q_trajectory=None, time=None):

        # self manual calibration of self local dependency is no longer used due to redundancy reason.
        # Robot coordinates
        # CoM_x, CoM_y, phi = q[0], q[1], q[2]
        # theta1, theta2, theta3, theta4 = q[3], q[4], q[5], q[6]

        fk = forward_kinematics(q)

        # === Update body polygon ===
        cx, cy, phi = fk.CoM_x, fk.CoM_y, q[2]
        L = self.robot.get_L()
        h = self.robot.h

        # Define local corners (rectangle centered at origin)
        half_L = L / 2
        half_h = h / 2
        corners = np.array([
            [-half_L, -half_h],
            [ half_L, -half_h],
            [ half_L,  half_h],
            [-half_L,  half_h]
        ])

        # Rotate and translate to CoM
        rot = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi),  np.cos(phi)]
        ])
        rotated_corners = corners @ rot.T + np.array([cx, cy])

        # Update polygon shape (just self line is enough)
        self.body_patch.set_xy(rotated_corners)

        self.com_plot.set_data([fk.CoM_x], [fk.CoM_y])

        self.shoulder_plots.set_data(
            [fk.front_shoulder_x, fk.rear_shoulder_x],
            [fk.front_shoulder_y, fk.rear_shoulder_y]
        )

        self.knee_plots.set_data(
            [fk.front_knee_x, fk.rear_knee_x],
            [fk.front_knee_y, fk.rear_knee_y]
        )

        self.ankle_plots.set_data(
            [fk.front_ankle_x, fk.rear_ankle_x],
            [fk.front_ankle_y, fk.rear_ankle_y]
        )
        # Shoulder → Knee → Ankle for both legs
        self.link_lines_front.set_data(
            [fk.front_shoulder_x, fk.front_knee_x, fk.front_ankle_x],
            [fk.front_shoulder_y, fk.front_knee_y, fk.front_ankle_y]
        )

        self.link_lines_rear.set_data(
            [fk.rear_shoulder_x, fk.rear_knee_x, fk.rear_ankle_x],
            [fk.rear_shoulder_y, fk.rear_knee_y, fk.rear_ankle_y]
        )

        # if u is not None:

        #     force_scale = 1/300
        #     GRF1_x, GRF1_y, GRF2_x, GRF2_y = u[:4] * force_scale

        #     if grf:
        #         self.force_lines_front.set_data(
        #             [fk.front_ankle_x, fk.front_ankle_x + GRF1_x],
        #             [fk.front_ankle_y, fk.front_ankle_y + GRF1_y]
        #         )

        #         self.force_lines_rear.set_data(
        #             [fk.rear_ankle_x, fk.rear_ankle_x + GRF2_x],
        #             [fk.rear_ankle_y, fk.rear_ankle_y + GRF2_y]
        #         )

        # if q_trajectory is not None:
        #     x_traj, y_traj = q_trajectory[0], q_trajectory[1]
            
        #     if trajectory_line:
        #         self.q_trajectory.set_data(x_traj, y_traj)
        
        # if timer: 
        #     if time is not None:
        #         self.timer_text.set_text(f"Time: {time:.2f} s")

        plt.draw()


# def user_output_preferences(show_grf_output):
#     # Output for the ground reaction forces
#     if show_grf_output:
#         print("GRF over time:")

#         for i in range (N - 1):
#             t = time_array[i]
#             F1x = U_sol[0, i]
#             F1y = U_sol[1, i]
#             F2x = U_sol[2, i]
#             F2y = U_sol[3, i]

#             print(f"Time: {t:0.3f}s - F1x: {F1x: .3f}N, F1y: {F1y: .3f}N, F2x: {F2x: .3f}N, F2y: {F2y: .3f}N")
