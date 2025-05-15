# visualization.py


from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from kinematics import forward_kinematics
from robot import h, L, leg_length


class QuadrupedSimulator:
    def __init__(this, L, leg_length, figax=None):
        this.L = L
        this.leg_length = leg_length
        this.h = h

        # Initialize the plot and axis
        if figax is None:
            this.fig, this.ax = plt.subplots()
        else:
            this.fig, this.ax = figax
        
        # Set plot appearance such size and orientn.
        this.ax.set_aspect("equal")
        this.ax.set_xlim(-2.0, 4.5)
        this.ax.set_ylim(-1.0, 4.0)

        # Rectangle for body
        this.body_patch = patches.Polygon([[0, 0], [0, 0], [0, 0], [0, 0]], closed=True, color="#5FB257", alpha=0.5)
        this.ax.add_patch(this.body_patch)

        # CoM marker
        (this.com_plot,) = this.ax.plot([], [], "ko", markersize=3, label="CoM")

        # Joint markers
        (this.shoulder_plots,) = this.ax.plot([], [], "o", markersize=3, markerfacecolor='white', markeredgecolor='blue')
        (this.knee_plots,) = this.ax.plot([], [], "o", markersize=3, markerfacecolor='white', markeredgecolor='blue')
        (this.ankle_plots,) = this.ax.plot([], [], "o", markersize=3, markerfacecolor='white', markeredgecolor='blue')

        # Leg lines
        (this.link_lines_front,) = this.ax.plot([], [], "k-", linewidth=2)
        (this.link_lines_rear,) = this.ax.plot([], [], "k-", linewidth=2)

        # # GRFs
        # if grf:
        #     (this.force_lines_front,) = this.ax.plot([], [], "r-", linewidth=1)
        #     (this.force_lines_rear,) = this.ax.plot([], [], "r-", linewidth=1)

        # # Trajectory line
        # if trajectory_line:
        #     (this.q_trajectory,) = this.ax.plot([], [], "k--", lw=1, alpha=0.5)

        # Draw ground line
        this.ax.plot([-3.0, 5.0], [0, 0], "k-", linewidth=1)

        # # Timer
        # if timer:
        #     this.timer_text = this.ax.text(
        #         0.5, 1.02,                  # x and y in axis coordinates (top-center, a bit above)
        #         "Time: 0.00 s",             # Initial text
        #         transform=this.ax.transAxes,
        #         ha="center", va="bottom", fontsize=10, color="black", fontweight="bold"
        #     )

        # if fixed_ground: 
        #     x_start = -3.0
        #     x_end = 5.0
        #     spacing = 0.2
        #     length = 0.2
        #     x_vals = np.arange(x_start, x_end, spacing)
        #     for x in x_vals:
        #         this.ax.plot([x + length, x], [-0.01, -0.05 - length], "k-", linewidth=1)

    def set_data(this, q, u=None, q_trajectory=None, time=None):

        # This manual calibration of this local dependency is no longer used due to redundancy reason.
        # Robot coordinates
        # CoM_x, CoM_y, phi = q[0], q[1], q[2]
        # theta1, theta2, theta3, theta4 = q[3], q[4], q[5], q[6]

        fk = forward_kinematics(q)

        # === Update body polygon ===
        cx, cy, phi = fk.CoM_x, fk.CoM_y, q[2]
        L = this.L
        h = this.h

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

        # Update polygon shape (just this line is enough)
        this.body_patch.set_xy(rotated_corners)

        this.com_plot.set_data([fk.CoM_x], [fk.CoM_y])

        this.shoulder_plots.set_data(
            [fk.front_shoulder_x, fk.rear_shoulder_x],
            [fk.front_shoulder_y, fk.rear_shoulder_y]
        )

        this.knee_plots.set_data(
            [fk.front_knee_x, fk.rear_knee_x],
            [fk.front_knee_y, fk.rear_knee_y]
        )

        this.ankle_plots.set_data(
            [fk.front_ankle_x, fk.rear_ankle_x],
            [fk.front_ankle_y, fk.rear_ankle_y]
        )
        # Shoulder → Knee → Ankle for both legs
        this.link_lines_front.set_data(
            [fk.front_shoulder_x, fk.front_knee_x, fk.front_ankle_x],
            [fk.front_shoulder_y, fk.front_knee_y, fk.front_ankle_y]
        )

        this.link_lines_rear.set_data(
            [fk.rear_shoulder_x, fk.rear_knee_x, fk.rear_ankle_x],
            [fk.rear_shoulder_y, fk.rear_knee_y, fk.rear_ankle_y]
        )

        # if u is not None:

        #     force_scale = 1/300
        #     GRF1_x, GRF1_y, GRF2_x, GRF2_y = u[:4] * force_scale

        #     if grf:
        #         this.force_lines_front.set_data(
        #             [fk.front_ankle_x, fk.front_ankle_x + GRF1_x],
        #             [fk.front_ankle_y, fk.front_ankle_y + GRF1_y]
        #         )

        #         this.force_lines_rear.set_data(
        #             [fk.rear_ankle_x, fk.rear_ankle_x + GRF2_x],
        #             [fk.rear_ankle_y, fk.rear_ankle_y + GRF2_y]
        #         )

        # if q_trajectory is not None:
        #     x_traj, y_traj = q_trajectory[0], q_trajectory[1]
            
        #     if trajectory_line:
        #         this.q_trajectory.set_data(x_traj, y_traj)
        
        # if timer: 
        #     if time is not None:
        #         this.timer_text.set_text(f"Time: {time:.2f} s")

        plt.draw()


