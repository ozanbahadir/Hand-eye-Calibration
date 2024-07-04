import os
import time
import pdb
import pybullet as p
import pybullet_data
import utils_ur5_robotiq140
from collections import deque
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pickle

# Get the current directory path
mypath = Path().absolute()

# Set the server mode (GUI or DIRECT)
serverMode = p.GUI

# Path to the URDF file for the UR5 robot with Robotiq 140 gripper
sisbotUrdfPath = "./urdf/ur5_robotiq_140.urdf"

# Connect to the PyBullet physics server
physicsClient = p.connect(serverMode)

# Set the additional search path for loading URDF files
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Define the start positions and orientations for objects in the environment
deskStartPos = [0.1, -0.49, 0.85]
deskStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
tableStartPos = [0.0, -0.9, 0.8]
tableStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
ur5standStartPos = [-0.7, -0.36, 0.0]
ur5standStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

# Load the URDF files for the objects
boxId1 = p.loadURDF("./urdf/objects/table.urdf", tableStartPos, tableStartOrientation, useFixedBase=True)
boxId1 = p.loadURDF("./urdf/objects/ur5_stand.urdf", ur5standStartPos, ur5standStartOrientation, useFixedBase=True)

# Functions to save RGB and RGBD images
def save_rgb_image(img, c):
    file = open(str(mypath) + "/Images/rgb_image/" + "rgb_image%s.pkl" % c, 'wb')
    pickle.dump(np.asarray(img), file)
    file.close()

def save_rgbd_image(img, c):
    file = open(str(mypath) + "/Images/rgb_image/" + "rgbd_image%s.pkl" % c, 'wb')
    pickle.dump(np.asarray(img), file)
    file.close()

# Functions to print data to files
def print_data_image(a, target, cref, campose, camorn, Resultfilez, Resultfilez1, Resultfilez2):
    for i in a:
        Resultfilez.write(str(i) + "\t")
    for i in cref:
        Resultfilez.write(str(i) + "\t")
    Resultfilez.write("\n")
    for i in campose:
        Resultfilez2.write(str(i) + "\t")
    for i in camorn:
        Resultfilez2.write(str(i) + "\t")
    Resultfilez2.write("\n")
    for i in target:
        Resultfilez1.write(str(i) + "\t")
    Resultfilez1.write("\n")

def print_data_image_multiple(a, target, cref, campose, camorn, Resultfilez, Resultfilez1, Resultfilez2):
    for index, i in enumerate(a):
        Resultfilez.write(str(i) + "\t")
    for i in cref:
        Resultfilez.write(str(i) + "\t")
    Resultfilez.write("\n")
    for i in campose:
        Resultfilez2.write(str(i) + "\t")
    for i in camorn:
        Resultfilez2.write(str(i) + "\t")
    Resultfilez2.write("\n")
    for i in target:
        Resultfilez1.write(str(i) + "\t")
    Resultfilez1.write("\n")

# Define camera image parameters
width = 1024
height = 1024
fov = 40
aspect = width / height
near = 0.2
far = 2
view_matrix = p.computeViewMatrix([0.0, 1.5, 0.5], [0, 0, 0.7], [0, 1, 0])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Setup UR5 robot with Robotiq 140 gripper
robotStartPos = [0, 0, 0.0]
robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
print("----------------------------------------")
print("Loading robot from {}".format(sisbotUrdfPath))
robotID = p.loadURDF(sisbotUrdfPath, robotStartPos, robotStartOrn, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
joints, controlRobotiqC2, controlJoints, mimicParentName = utils_ur5_robotiq140.setup_sisbot(p, robotID)
eefID = 7  # end-effector link

# Set damping for robot arm and gripper
jd = [0.1] * 12
jd = jd * 0
np.random.seed(111)
sample_number = 100

# Define ranges for random sampling of positions and orientations
ranges = np.array([[-1.6, 1.6], [-1.9, 0.0], [0.9, 1.8]])
ranges1 = np.array([[-math.pi / 2, math.pi / 2], [-math.pi / 2, math.pi / 2], [-math.pi / 2, math.pi / 2]])

# Generate random positions and orientations
Rpose = np.random.uniform(ranges[:, 0], ranges[:, 1], size=(sample_number, ranges.shape[0]))
R_orientation = np.random.uniform(ranges1[:, 0], ranges1[:, 1], size=(sample_number, ranges.shape[0]))

# Camera settings
userParams = dict()
yaw = 0
pitch = -50
roll = 0
upAxisIndex = 2
camDistance = 4
near = 0.01
far = 100
fov = 60
aspect = width / height
look = [0.0, 0.0, 0.3]
distance = 1.
camDistance = 2.8
c = 1

# Add user debug parameters for Cartesian space and gripper control
xin = p.addUserDebugParameter("x", -3.14, 3.14, 0.11)
yin = p.addUserDebugParameter("y", -3.14, 3.14, -0.49)
zin = p.addUserDebugParameter("z", 0.9, 1.3, 1.29)
rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, 1.57)
yawId = p.addUserDebugParameter("yaw", -3.14, 3.14, -1.57)
gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.085)

# Open files to save results
Resultfilez = open(str(mypath) + "/Images/" + "input.txt1", "a")
Resultfilez1 = open(str(mypath) + "/Images/" + "target_pixel1.txt", "a")
Resultfilez2 = open(str(mypath) + "/Images/" + "camera_pose1.txt", "a")

# Create a mapping of link names to indices
_link_name_to_index = {p.getBodyInfo(robotID)[0].decode('UTF-8'): -1,}
for _id in range(p.getNumJoints(robotID)):
    _name = p.getJointInfo(robotID, _id)[12].decode('UTF-8')
    _link_name_to_index[_name] = _id
print(_link_name_to_index)

# Iterate through different pitch and yaw values
# Iterate through different pitch and yaw values
for pitch in range(-180, -180, 10):
    for yaw in range(0, 10, 10):
        for index, u in enumerate(Rpose):
            # Get depth values using the OpenGL renderer
            projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
            control_cnt = 0
            while control_cnt < 100:
                x = Rpose[index][0]
                y = Rpose[index][1]
                z = Rpose[index][2]
                orn = p.getQuaternionFromEuler([R_orientation[index][0], R_orientation[index][1], R_orientation[index][2]])

                view_matrix = p.computeViewMatrixFromYawPitchRoll(look, camDistance, yaw, pitch, roll, upAxisIndex)
                gripper_opening_length = p.readUserDebugParameter(gripper_opening_length_control)
                gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation

                # Apply inverse kinematics to get joint positions
                jointPose = p.calculateInverseKinematics(robotID, eefID, [x, y, z], orn, jointDamping=jd)
                for i, name in enumerate(controlJoints):
                    pose = jointPose[i]
                    joint = joints[name]

                    if name == mimicParentName:
                        controlRobotiqC2(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
                    else:
                        p.setJointMotorControl2(robotID, joint.id, p.POSITION_CONTROL,
                                                targetPosition=pose, force=joint.maxForce,
                                                maxVelocity=joint.maxVelocity)

                control_cnt += 1
                p.stepSimulation()

            # Get camera position and orientation
            cameraPos = [-view_matrix[12], -view_matrix[13], -view_matrix[14]]
            cameraOrnEuler = [roll, pitch, yaw]
            cameraOrnEuler = np.radians(cameraOrnEuler)
            cameraOrnQuaternion = p.getQuaternionFromEuler(cameraOrnEuler)

            time.sleep(10)
            img_arr = p.getCameraImage(width, height, view_matrix, projection_matrix, [0, 1, 0], renderer=p.ER_BULLET_HARDWARE_OPENGL)

            control_cnt += 1
            rXYZ = p.getLinkState(robotID, 15)[0]  # Get the real position of the end-effector
            rxyzw = p.getLinkState(robotID, 15)[1]  # Get the real orientation of the end-effector

            rroll, rpitch, ryaw = p.getEulerFromQuaternion(rxyzw)
            cam2ref = p.multiplyTransforms(cameraPos, cameraOrnQuaternion, rXYZ, rxyzw)
            cam2ref1 = [cam2ref[0][0], cam2ref[0][1], cam2ref[0][2], cam2ref[1][0], cam2ref[1][1], cam2ref[1][2], cam2ref[1][3]]

            # Calculate screen coordinates
            b = np.matmul(np.asarray(projection_matrix).reshape(4, 4).T, np.asarray(view_matrix).reshape(4, 4).T)
            twod = np.matmul(b, [rXYZ[0], rXYZ[1], rXYZ[2], 1])
            twod /= twod[3]
            scrn_co_x = (twod[0] + 1) / 2 * width
            scrn_co_y = (twod[1] + 1) / 2 * height
            scrn_co_y = height - scrn_co_y

            if 0 <= scrn_co_x < width and 0 <= scrn_co_y < height:
                print_data_image([rXYZ[0], rXYZ[1], rXYZ[2], rxyzw[0], rxyzw[1], rxyzw[2], rxyzw[3]],
                                 [scrn_co_x, scrn_co_y], cam2ref1, cameraPos, cameraOrnQuaternion,
                                 Resultfilez, Resultfilez1, Resultfilez2)

                # Save RGB and RGBD images
                rgb = img_arr[2]
                np_img_arr = np.reshape(rgb, (width, height, 4))
                np_img_arr = np_img_arr * (1. / 255.)
                depth_buffer_opengl = np.reshape(img_arr[3], [width, height])
                depthImg = far * near / (far - (far - near) * depth_buffer_opengl)
                save_rgb_image(np_img_arr, c)
                save_rgbd_image(depthImg, c)
                c += 1

            time.sleep(1)

# Close result files
Resultfilez.close()
Resultfilez1.close()
Resultfilez2.close()

# Disconnect from the PyBullet server
p.disconnect()
