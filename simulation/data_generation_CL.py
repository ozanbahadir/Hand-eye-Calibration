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


mypath = Path().absolute()
serverMode = p.GUI # GUI/DIRECT
sisbotUrdfPath = "./urdf/ur5_robotiq_140.urdf"

# connect to engine servers
physicsClient = p.connect(serverMode)
# add search path for loadURDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#p.getCameraImage(640,480)

# define world
#p.setGravity(0,0,-10) # NOTE
#planeID = p.loadURDF("plane.urdf")

# define environment
deskStartPos = [0.1, -0.49, 0.85]
deskStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
#boxId = p.loadURDF("./urdf/objects/block.urdf", deskStartPos, deskStartOrientation)

tableStartPos = [0.0, -0.9, 0.8]
tableStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId1 = p.loadURDF("./urdf/objects/table.urdf", tableStartPos, tableStartOrientation,useFixedBase = True)

ur5standStartPos = [-0.7, -0.36, 0.0]
ur5standStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId1 = p.loadURDF("./urdf/objects/ur5_stand.urdf", ur5standStartPos, ur5standStartOrientation,useFixedBase = True)


def save_rgb_image(img,c):
    file = open(str(mypath) +"/CalibrationwithImages_H2/rgb_image/"+"rgb_image%s.pkl" % c, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(np.asarray(img), file)
    file.close()
    #imarray = np.asarray(img)
    #im1 = Image.fromarray(imarray.astype('uint16'))
    #im1.save(str(mypath) +"/CalibrationwithImages_2/rgb_image/"+"rgb_image%s.png" % c)
    #plt.imshow(np.asarray(img))
    #plt.savefig(str(mypath) +"/CalibrationwithImages_3/rgb_image/"+"rgb_image%s.png" % c)

def save_rgbd_image(img,c):
    file = open(str(mypath) +"/CalibrationwithImages_H2/rgbd_image/"+"rgbd_image%s.pkl" % c, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(np.asarray(img), file)
    file.close()
    #plt.imshow(np.asarray(img))
    #plt.savefig(str(mypath) +"/CalibrationwithImages_3/rgbd_image/"+"rgbd_image%s.png" % c)

def print_data_image(a,target,cref,campose,camorn,Resultfilez,Resultfilez1,Resultfilez2):
    for i in a:
       Resultfilez.write(str(i)+"\t")
    for i in cref:
       Resultfilez.write(str(i)+"\t")
    Resultfilez.write("\n")
    for i in campose:
       Resultfilez2.write(str(i)+"\t")
    for i in camorn:
       Resultfilez2.write(str(i)+"\t")

    Resultfilez2.write("\n")
    for i in target:
       Resultfilez1.write(str(i)+"\t")
    Resultfilez1.write("\n")










# define camera image parameter

width = 1024
height = 1024
fov = 40
aspect = width / height
near = 0.2
far = 2
view_matrix = p.computeViewMatrix([0.0, 1.5, 0.5], [0, 0, 0.7], [0, 1, 0])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)


# setup ur5 with robotiq 140
robotStartPos = [0,0,0.0]
robotStartOrn = p.getQuaternionFromEuler([0,0,0])
print("----------------------------------------")
print("Loading robot from {}".format(sisbotUrdfPath))
robotID = p.loadURDF(sisbotUrdfPath, robotStartPos, robotStartOrn,useFixedBase = True,
                     flags=p.URDF_USE_INERTIA_FROM_FILE)
joints, controlRobotiqC2, controlJoints, mimicParentName = utils_ur5_robotiq140.setup_sisbot(p, robotID)
eefID = 7 # ee_link

# start simulation
ABSE = lambda a,b: abs(a-b)

# set damping for robot arm and gripper
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
jd = jd*0
np.random.seed(111)
sample_number=700
ranges = np.array([[-1.6, 1.6],[-1.9, 0.0], [0.9, 1.8]])
ranges1 = np.array([[ -math.pi/2 , math.pi/2],[ -math.pi/2, math.pi/2], [ -math.pi/2, math.pi/2]])

Rpose=np.random.uniform(ranges[:, 0], ranges[:, 1],size=(sample_number, ranges.shape[0]))
R_orientation= np.random.uniform(ranges1[:, 0], ranges1[:, 1],size=(sample_number, ranges.shape[0]))
userParams = dict()
yaw = 0
pitch = -50

roll=0
upAxisIndex = 2
camDistance = 4
near = 0.01
far = 100
fov = 60

#fov = 40
aspect = width / height
#near = 0.2
far = 100
look = [0.0,0.0,0.3]
distance = 1.
camDistance = 2.8
c=1

# custom sliders to tune parameters (name of the parameter,range,initial value)
# Task space (Cartesian space)
xin = p.addUserDebugParameter("x", -3.14, 3.14, 0.11)
yin = p.addUserDebugParameter("y", -3.14, 3.14, -0.49)
zin = p.addUserDebugParameter("z", 0.9, 1.3, 1.29)
rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0) #-1.57 yaw
pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, 1.57)
yawId = p.addUserDebugParameter("yaw", -3.14, 3.14, -1.57) # -3.14 pitch
gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length",0,0.085,0.085)

# Joint space 
#userParams[0] = p.addUserDebugParameter("shoulder_pan_joint", -3.14, 3.14, -1.57)
#userParams[1] = p.addUserDebugParameter("shoulder_lift_joint", -3.14, 3.14, -1.57)
#userParams[2] = p.addUserDebugParameter("elbow_joint", -3.14, 3.14, 1.57)
#userParams[3] = p.addUserDebugParameter("wrist_1_joint",-3.14, 3.14, -1.57)
#userParams[4] = p.addUserDebugParameter("wrist_2_joint", -3.14, 3.14, -1.57)
#userParams[5] = p.addUserDebugParameter("wrist_3_joint", -3.14, 3.14, 0)   

#    # Camera parameter for computeViewMatrix (see the pybullet document)
#c1 = p.addUserDebugParameter("cc1", -3, 5.5, 0.132)
#c2 = p.addUserDebugParameter("cc2", -3, 5.5, -1.524)
#c3 = p.addUserDebugParameter("cc3", -3, 5.5, 1.205)
#c4 = p.addUserDebugParameter("cc4", -3, 5.5, 0.132)
#c5 = p.addUserDebugParameter("cc5", -3, 5.5, -0.539)
#c6 = p.addUserDebugParameter("cc6", -3, 5.5, 1.116)

control_cnt = 0;
Resultfilez = open(str(mypath) +"/CalibrationwithImages_H2/"+"input.txt","a")
Resultfilez1 = open(str(mypath) +"/CalibrationwithImages_H2/"+"target_pixel.txt","a")
Resultfilez2 = open(str(mypath) +"/CalibrationwithImages_H2/"+"camera_pose.txt","a")

_link_name_to_index = {p.getBodyInfo(robotID)[0].decode('UTF-8'):-1,}
for _id in range(p.getNumJoints(robotID)):
	_name = p.getJointInfo(robotID, _id)[12].decode('UTF-8')
	_link_name_to_index[_name] = _id
print(_link_name_to_index)
#time.sleep(222222)
    
#while(1):
#def run(u,Rpose,R_orientation):
#for pitch in range (-50,10,10) :
#    for yaw in range (0,180,10) :
for pitch in range (-50,-40,10) :
    for yaw in range (0,10,10) :
        for index,u in enumerate(Rpose):
       

# Get depth values using the OpenGL renderer
            projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    #images = p.getCameraImage(width,height,view_matrix,projection_matrix,shadow=True,renderer=p.ER_BULLET_HARDWARE_OPENGL)
    #rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.
    
            control_cnt = 0;
            while(control_cnt<100):
                x = Rpose[index][0]#Rpose[control_cnt][0]#p.readUserDebugParameter(xin)
                y = Rpose[index][1]#Rpose[control_cnt][1]#p.readUserDebugParameter(yin) 
                z = Rpose[index][2]#Rpose[control_cnt][2]#p.readUserDebugParameter(zin)
		#        roll = p.readUserDebugParameter(rollId)
		#        pitch = p.readUserDebugParameter(pitchId)
		#        yaw = p.readUserDebugParameter(yawId)
                orn = p.getQuaternionFromEuler([R_orientation[index][0],R_orientation[index][1],R_orientation[index][2] ])

		# read the value of camera parameter
		#        cc1 = p.readUserDebugParameter(c1)
		#        cc2 = p.readUserDebugParameter(c2)
		#        cc3 = p.readUserDebugParameter(c3)
		#        cc4 = p.readUserDebugParameter(c4)
		#        cc5 = p.readUserDebugParameter(c5)
		#        cc6 = p.readUserDebugParameter(c6)
		#view_matrix = p.computeViewMatrix([cc1, cc2, cc3], [cc4, cc5, cc6], [0, 1, 0])
                view_matrix = p.computeViewMatrixFromYawPitchRoll(look, camDistance, yaw, pitch, roll, upAxisIndex)
#                tcameraPos=[-view_matrix[12],-view_matrix[13],-view_matrix[14]]
#                tcameraOrnEuler=[roll,pitch,yaw]
#                tcameraOrnEuler=np.radians(tcameraOrnEuler)
#                tcameraOrnQuaternion=p.getQuaternionFromEuler(tcameraOrnEuler)

		#print (tcameraPos)
		#print(tcameraOrnQuaternion)
		#time.sleep(55555)
                gripper_opening_length = p.readUserDebugParameter(gripper_opening_length_control)
                gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)    # angle calculation

		# apply IK
                jointPose = p.calculateInverseKinematics(robotID, eefID, [x,y,z],orn,jointDamping=jd)
                for i, name in enumerate(controlJoints):
                    pose = jointPose[i]
                    joint = joints[name]
		
		
		       # if i != 6:
			   # pose1 = p.readUserDebugParameter(userParams[i])
                    if name==mimicParentName:
                        controlRobotiqC2(controlMode=p.POSITION_CONTROL,targetPosition=gripper_opening_angle)
                    else:
                         p.setJointMotorControl2(robotID, joint.id, p.POSITION_CONTROL,
						        targetPosition=pose, force=joint.maxForce, 
						        maxVelocity=joint.maxVelocity)

		#            if control_cnt < 100:
		#		    # control robot joints
		#                p.setJointMotorControl2(robotID, joint.id, p.POSITION_CONTROL,
		#		                        targetPosition=pose, force=joint.maxForce, 
		#		                        maxVelocity=joint.maxVelocity)
		#            else:
		#                p.setJointMotorControl2(robotID, joint.id, p.POSITION_CONTROL,targetPosition=pose, force=joint.maxForce,maxVelocity=joint.maxVelocity)
                control_cnt+=1
                p.stepSimulation()
            cameraPos=[-view_matrix[12],-view_matrix[13],-view_matrix[14]]
            cameraOrnEuler=[roll,pitch,yaw]
            cameraOrnEuler=np.radians(cameraOrnEuler)
            cameraOrnQuaternion=p.getQuaternionFromEuler(cameraOrnEuler)
	    #cam2ref = pybullet.multiplyTransforms(cameraPos, cameraOrnQuaternion,a[0], a[1])
            time.sleep(10)
            img_arr = p.getCameraImage(width, height, view_matrix,projection_matrix, [0,1,0],renderer=p.ER_BULLET_HARDWARE_OPENGL)#renderer=p.ER_TINY_RENDERER)    
            control_cnt = control_cnt + 1
	    #rXYZ = p.getLinkState(robotID, eefID)[0] # real X
	    #rxyzw = p.getLinkState(robotID, eefID)[1] # real rpy
            rXYZ = p.getLinkState(robotID, 15)[0] # real X
            rxyzw = p.getLinkState(robotID, 15)[1] # real rpy
	    #time.sleep(55555)
            rroll, rpitch, ryaw = p.getEulerFromQuaternion(rxyzw)
            cam2ref = p.multiplyTransforms(cameraPos, cameraOrnQuaternion,rXYZ, rxyzw)
	    #quatcam= p.getEulerFromQuaternion([cam2ref[1][0],cam2ref[1][1],cam2ref[1][2],cam2ref[1][3]])
            cam2ref1=[cam2ref[0][0],cam2ref[0][1],cam2ref[0][2],cam2ref[1][0],cam2ref[1][1],cam2ref[1][2],cam2ref[1][3]]
            b=np.matmul(np.asarray(projection_matrix).reshape(4,4).T, np.asarray(view_matrix).reshape(4,4).T) 
            twod=np.matmul(b,[rXYZ[0],rXYZ[1],rXYZ[2],1])
            twod/=twod[3]
            scrn_co_x = (twod[0] + 1) / 2 * width
            scrn_co_y = (twod[1]+ 1) / 2 * height
            scrn_co_y=height - scrn_co_y
            if scrn_co_x<width and scrn_co_x>=0 and scrn_co_y<height and scrn_co_y>=0:
		#print_data_image([rXYZ[0],rXYZ[1],rXYZ[2],rroll,rpitch,ryaw],[scrn_co_x,scrn_co_y],cam2ref1,Resultfilez,Resultfilez1)
                print_data_image([rXYZ[0],rXYZ[1],rXYZ[2],rxyzw[0],rxyzw[1],rxyzw[2],rxyzw[3]],[scrn_co_x,scrn_co_y],cam2ref1,cameraPos,cameraOrnQuaternion,Resultfilez,Resultfilez1,Resultfilez2)
		#img_arr = p.getCameraImage(width, height, view_matrix,projection_matrix, [0,1,0],renderer=p.ER_BULLET_HARDWARE_OPENGL)#renderer=p.ER_TINY_RENDERER)
                rgb=img_arr[2]
                np_img_arr = np.reshape(rgb, (width, height, 4)) 
                np_img_arr = np_img_arr * (1. / 255.)  
                depth_buffer_opengl = np.reshape(img_arr[3], [width, height]) 
		#seg=np.reshape(img_arr[4], [width, height])
		#print(seg)
		
		#plt.imshow(np.asarray(seg))
		#plt.savefig(str(mypath) +"/CalibrationwithImages_4/"+"seg.png" )
                depthImg = far * near / (far - (far - near) * depth_buffer_opengl)
                save_rgb_image(np_img_arr,c)
                save_rgbd_image(depthImg,c)
                c+=1
            time.sleep(1)
        #print("err_x= {:.2f}, err_y= {:.2f}, err_z= {:.2f}".format(*list(map(ABSE,[x,y,z],rXYZ))))
	    
	#	    print("err_x= {:.2f}, err_y= {:.2f}, err_z= {:.2f}".format(*list(map(ABSE,[x,y,z],rXYZ))))
	#	    print("err_r= {:.2f}, err_o= {:.2f}, err_y= {:.2f}".format(*list(map(ABSE,[roll,pitch,yaw],[rroll, rpitch, ryaw]))))
	#	    print("x_= {:.2f}, y= {:.2f}, z= {:.2f}".format(rXYZ[0],rXYZ[1],rXYZ[2]))
	#	    print("rroll_= {:.2f}, rpitch= {:.2f}, ryaw= {:.2f}".format(rroll,rpitch,ryaw))
	#		# current box coordinate
	#	    #cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
	#	    print(cubePos,cubeOrn)
		    
Resultfilez1.close()
Resultfilez.close()
Resultfilez2.close()



