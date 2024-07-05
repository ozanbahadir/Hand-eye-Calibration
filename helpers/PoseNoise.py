import numpy as np
import os
import quaternion
import torch

import torchgeometry as tgm



import mathutils

from mathutils import Quaternion
from mathutils import Vector
quat1 = Quaternion((1, 2, 3, 4))
quat1_norm = quat1.normalized()

print(quat1)
print(quat1_norm)

#os.chdir("/home/ozanba/anaconda3/envs/env1/lib/python3.6/site-packages")
import random
def noisyPose(pose,ns,ns2):
    #mean = 0
    #var = ns
    #sigma = var ** 0.5
    #sigma = var
    #sigma=np.std(image)
    pose1=pose.detach().cpu().numpy()
    #ns=np.deg2rad(ns)
    sigma=ns
    tens,ch=pose.shape

    if ch==7:
        gauss1 = np.random.normal(0, sigma, (tens, 3))
        gaussOrn = modify(pose.narrow(1, 3, 4), ns2)
    else:
        gauss1 = np.random.normal(0, sigma, (tens, 3))
        gauss2 = np.random.normal(0, sigma, (tens, 3))
        gaussOrn1 = modify(pose.narrow(1, 3, 4), ns2)
        gaussOrn2 = modify(pose.narrow(1, 10, 4), ns2)

    gauss1 = np.random.normal(0, sigma, (tens,3))
    #gauss = np.random.normal(0, sigma, (4,3))
    gaussOrn=modify(pose.narrow(1, 3, 4),ns2)
    for index,i in enumerate(pose1):
        for jindex,j in enumerate(i):
            if len(i)==7:
                if jindex<3:
                    pose1[index][jindex]+=gauss1[index][jindex]
                else:
                    pose1[index][jindex]=gaussOrn[index][jindex-3]
            else:
                if jindex<3:
                    pose1[index][jindex]+=gauss1[index][jindex]
                elif jindex>=3 and jindex<7:
                    pose1[index][jindex]=gaussOrn1[index][jindex-3]
                elif jindex>=7 and jindex<10:
                    pose1[index][jindex] += gauss2[index][jindex-7]
                else:
                    pose1[index][jindex] = gaussOrn2[index][jindex - 10]


    noisy = torch.tensor(pose1)
    return noisy

def modify(orn,ns2):
    ornNd=orn.detach().cpu().numpy()
    #ornQuat = quaternion.as_quat_array(ornNd)
    ornQuat=[Quaternion(i).normalized() for i in ornNd]
    #ornQuat123=np.asarray(ornQuat)
    #print(ornQuat[0].magnitude)
    gauss1 =np.random.normal(0, ns2, (len(ornNd),4))
    gauss1Q=[Quaternion(i).normalized() for i in gauss1]

    #gauss1Q = [Quaternion(i).normalized() for i in gauss1]
    #gauss1Q123=np.asarray(gauss1Q)
    #print(gauss1Q[0].magnitude)
    #gauss1Q = quaternion.as_quat_array(gauss1)



    gauss=[Quaternion.cross( gauss1Q[i],ornQuat[i]).normalized() for i in range(len(gauss1Q))]

    #gauss123=ornQuat*gauss1Q
    #gaussNd=quaternion.as_float_array(gauss)
    #print(gauss[0].magnitude)
    gaussNd=np.asarray(gauss)
    #print(gauss)
    #gauss33=quaternion.quaternion(gauss)
    #print(gauss33)
    #print(gaussNd)
    return gaussNd

'''
angle_axis = torch.from_numpy(np.asarray([50.0,0.0,0.0]))
angle_axis2 = torch.from_numpy(np.asarray([0.0,0.0,50.0]))
#torch.rand(2, 3)  # Nx4
quaternion55 = tgm.angle_axis_to_quaternion(angle_axis)
quaternion50 = tgm.angle_axis_to_quaternion(angle_axis2) # N
quaternion60=quaternion55*quaternion50
print(quaternion60)
rot =np.array([[50.0,0.0,0.0],[50.0,0.0,0.0]])
rot = np.array(rot, copy=False)
quats = np.zeros(rot.shape[:-1]+(4,))
quats[..., 1:] = rot[...]/2
quats = quaternion.as_quat_array(quats)



rot2=np.array([[0.0,0.0,50.0],[0.0,0.0,50.0]])
rot2 = np.array(rot2, copy=False)
quats2 = np.zeros(rot2.shape[:-1]+(4,))
quats2[..., 1:] = rot2[...]/2
quats2 = quaternion.as_quat_array(quats2)
p=np.exp(quats)
p1=np.exp(quats2)
#h=np.array([p(3),p(1),p(2),p(0)])
#h=quaternion(p[3],p[1],p[2],p[0])
h=quaternion.as_float_array(p)
h1=quaternion.as_float_array(p1)
p3=p*p1
print(p3)

h10=quaternion.as_float_array(2*np.log(np.normalized(p)))[..., 1:]
h11=quaternion.as_float_array(2*np.log(np.normalized(p1)))[..., 1:]


h3=h10*h11

h20=quaternion.as_float_array(p1)
#rot2=np.array([0.0,0.0,50.0])
rot20 = np.array(h10, copy=False)
quats20 = np.zeros(h10.shape[:-1]+(4,))
quats20[..., 1:] = h10[...]/2
quats20 = quaternion.as_quat_array(quats20)
ppp=np.exp(quats20)

rot30 = np.array(h11, copy=False)
quats30 = np.zeros(h11.shape[:-1]+(4,))
quats30[..., 1:] = h11[...]/2
quats30 = quaternion.as_quat_array(quats30)
ppp1=np.exp(quats30)

print(p)
print(p1)
print(h)
print(h1)
print(p3)
print(h10)
print(h11)
print(ppp)
print(ppp1)
print(h20)

#h15=[h20[1],h20[2],h20[3],h20[0]]
#print(h15)
'''