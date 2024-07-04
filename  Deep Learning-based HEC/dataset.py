from pathlib import Path
from PIL import Image
import pickle
import cv2
import numpy as np
from skimage import data, io, filters
import matplotlib.pyplot as plt
mypath = Path().absolute()

def noisy(image,ns):
    mean = 0
    var = ns
    #sigma = var ** 0.5
    std=np.std(image)
    v=(std)**2
    row,col= image.shape
    gauss = np.random.normal(mean, v, (row, col))
    gauss = gauss.reshape(row, col)

    noisy = image + gauss
    plt.subplot(121), plt.imshow(image), plt.title('Origin')
    plt.subplot(122), plt.imshow(noisy), plt.title('Gaussian')
    plt.show()
    b=55
    return noisy
def noisy1(image,ns):
    #mean = 0
    #var = ns
    #sigma = var ** 0.5
    #sigma = var
    #sigma=np.std(image)
    #batch = image.view(image.size(0), image.size(1), -1)
    #std=batch.std(2).sum(0)
    std = np.std(image)
    ns=1
    #std = torch.std(batch, dim=2)
    #std1=std.cpu().detach().numpy()
    v = (ns*std)**2

    #gauss=[np.random.normal(0, v[i], (row, col)) for i in v]
    #gauss = np.random.normal(0, v, (row, col, ch))




    row,col,ch= image.shape
    gauss = np.random.normal(0, v, (row, col,ch))
    gauss1 = np.random.normal(0, v, (row, col, 3))
    #gauss1=np.concatenate((gauss[0], gauss[1], gauss[2]), axis = 3)
    #gauss = np.random.normal(0, v, (tens, ch, row, col))
    #gauss = np.random.normal(mean, sigma, (tens,ch,row, col))
    #gauss = gauss.reshape(ch,row,col)


    noisy = image + gauss
    noise=noisy-image
    #noise1 = np.zeros([256,256,3],dtype=np.uint8)
    #noise1.fill(255) # or img[:] = 255
    plt.figure(dpi=300)
    # plt.figure(figsize=(20,20))
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    plt.subplot(131), plt.imshow(image), plt.title('Original')
    plt.subplot(132), plt.imshow(gauss1), plt.title('Gaussian(100%)')
    plt.subplot(133), plt.imshow(noisy), plt.title('Image with Gaussian(100%)')
    plt.savefig("NoiseAnalysis.png")
    plt.show()

    return noisy

def read_image():
    datar = list()
    datas=list()
    #external=list()
    for i in range(696):#range(1644):
        #external.append([0,0.1928362,3.0298137,0.0,-0.4226183,0.0,0.906307787])
        pklr = open(str(mypath) + "/CalibrationwithImages_H2/rgb_image/rgb_image%s.pkl" %(i+1), 'rb')
        imgr = pickle.load(pklr)
        #frame = np.asarray(imgr, dtype=np.uint8)
        #imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY,dtype=np.uint8)

        size=(256,256)
        imgr=cv2.resize(imgr,size).astype(np.float32)
        #imgrN = noisy1(imgr, 1)
        #plt.imshow(imgr)
        #plt.imshow(imgrN)
        #plt.show()
        imgr=np.transpose(imgr, (2, 1, 0))
        #imgr=cv2.cvtColor(imgr[:,:,:3], cv2.COLOR_BGR2GRAY)
        #imgrN=noisy1(imgr[:3,:,:],0.0001)

        #plt.show()
        pkls = open(str(mypath) + "/CalibrationwithImages_H2/rgbd_image/rgbd_image%s.pkl" %(i+1), 'rb')
        imgs = pickle.load(pkls)
        imgs=cv2.resize(imgs,size)
        depth_max = np.nanmax(imgs)
        scale = 1.0 / depth_max
        depth_8bit = cv2.convertScaleAbs(imgs, None, scale).astype(np.float32)
        #imgrN = noisy(depth_8bit, 0.1)
        #plt.imshow(imgrN)
        #plt.show()
        imgs=np.expand_dims(depth_8bit, 0)
        #imgs = np.transpose(imgs, (2, 1, 0))
        #cv2.resize(img, size)
        datar.append(imgr[:3,:,:])
        datas.append(imgs)
    return datar,datas


def read_pixel():
    data = list()
    with open(str(mypath) + "/CalibrationwithImages_H2/target_pixel.txt",'r') as f:
        for line in f:
            linedata = line.split()

            temp=list()
            for index,j in enumerate(linedata):
                temp.append(float(j))
            data.append(temp)
    return data

def read_input():
    data = list()
    data1=list()
    with open(str(mypath) + "/CalibrationwithImages_H2/input.txt",'r') as f:
        for line in f:
            linedata = line.split()
            temp1=list()
            temp=list()
            for index,j in enumerate(linedata):
                if index<7:
                    temp.append(float(j))
                else:
                    temp1.append(float(j))

            data.append(temp)
            data1.append(temp1)
    return data,data1


def camera_pose():
    data = list()
    with open(str(mypath) + "/CalibrationwithImages_H2/camera_pose.txt",'r') as f:
        for line in f:
            linedata = line.split()

            temp=list()
            for index,j in enumerate(linedata):
                temp.append(float(j))
            data.append(temp)
    return data
def camera_pose1():
    data=list()
    for i in range(1644):
        data.append([0,0.1928362,3.0298137,0.0,-0.4226183,0.0,0.906307787])
    return data






