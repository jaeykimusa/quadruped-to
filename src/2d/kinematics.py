# kinematics.py
# fk.py


import numpy as np
from collections import namedtuple

from robot import L, leg_length


# kinematics output structure 
ForwardKinematicsPosition = namedtuple(
    "ForwardKinematicsPosition",
    [
        "CoM_x",
        "CoM_y",
        "front_shoulder_x",
        "front_shoulder_y",
        "front_knee_x",
        "front_knee_y",
        "front_ankle_x",
        "front_ankle_y",
        "rear_shoulder_x",
        "rear_shoulder_y",
        "rear_knee_x",
        "rear_knee_y",
        "rear_ankle_x",
        "rear_ankle_y",
    ], 
) 


# math_utils
def cos(theta):
    return np.cos(theta)


def sin(theta):
    return np.sin(theta)

def pi():
    return np.pi

def forward_kinematics(q):
    joint_positions = dict()
    joint_positions["CoM_x"] = q[0]
    joint_positions["CoM_y"] = q[1]

    phi = q[2]

    # Front leg
    theta1_front = q[3]
    theta2_front = q[4]

    front_shoulder_x = joint_positions["CoM_x"] + L / 2 * cos(phi)
    front_shoulder_y = joint_positions["CoM_y"] + L / 2 * sin(phi)

    front_knee_x = front_shoulder_x + leg_length * cos(phi - pi() / 2 + theta1_front)
    front_knee_y = front_shoulder_y + leg_length * sin(phi - pi() / 2 + theta1_front)

    front_ankle_x = front_knee_x + leg_length * cos(phi - pi() / 2 + theta1_front + theta2_front)
    front_ankle_y = front_knee_y + leg_length * sin(phi - pi() / 2 + theta1_front + theta2_front)

    joint_positions["front_shoulder_x"] = front_shoulder_x
    joint_positions["front_shoulder_y"] = front_shoulder_y
    joint_positions["front_knee_x"] = front_knee_x
    joint_positions["front_knee_y"] = front_knee_y
    joint_positions["front_ankle_x"] = front_ankle_x
    joint_positions["front_ankle_y"] = front_ankle_y

    # Rear leg
    theta1_rear = q[5]
    theta2_rear = q[6]

    rear_shoulder_x = joint_positions["CoM_x"] + -L / 2 * cos(phi)
    rear_shoulder_y = joint_positions["CoM_y"] + -L / 2 * sin(phi)

    rear_knee_x = rear_shoulder_x + leg_length * cos(phi - pi() / 2 + theta1_rear)
    rear_knee_y = rear_shoulder_y + leg_length * sin(phi - pi() / 2 + theta1_rear)

    rear_ankle_x = rear_knee_x + leg_length * cos(phi - pi() / 2 + theta1_rear + theta2_rear)
    rear_ankle_y = rear_knee_y + leg_length * sin(phi - pi() / 2 + theta1_rear + theta2_rear)

    joint_positions["rear_shoulder_x"] = rear_shoulder_x
    joint_positions["rear_shoulder_y"] = rear_shoulder_y
    joint_positions["rear_knee_x"] = rear_knee_x
    joint_positions["rear_knee_y"] = rear_knee_y
    joint_positions["rear_ankle_x"] = rear_ankle_x
    joint_positions["rear_ankle_y"] = rear_ankle_y

    return ForwardKinematicsPosition(**joint_positions)


# q = np.array([1,2,3,1,1,1,1])

# print(forward_kinematics(q))