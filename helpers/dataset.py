from pathlib import Path
from PIL import Image
import pickle
import cv2
import numpy as np
from skimage import data, io, filters
import matplotlib.pyplot as plt

# Get the absolute path of the current directory
mypath = Path().absolute()

def noisy(image, ns):
    """
    Adds Gaussian noise to a grayscale image.

    Args:
    image (numpy array): The input image.
    ns (float): Noise scale.

    Returns:
    numpy array: The noisy image.
    """
    mean = 0
    var = ns
    std = np.std(image)
    v = (std)**2
    row, col = image.shape
    gauss = np.random.normal(mean, v, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss

    # Display original and noisy images
    plt.subplot(121), plt.imshow(image), plt.title('Origin')
    plt.subplot(122), plt.imshow(noisy), plt.title('Gaussian')
    plt.show()
    
    return noisy

def noisy1(image, ns):
    """
    Adds Gaussian noise to a color image.

    Args:
    image (numpy array): The input image.
    ns (float): Noise scale.

    Returns:
    numpy array: The noisy image.
    """
    std = np.std(image)
    v = (ns * std)**2
    row, col, ch = image.shape
    gauss = np.random.normal(0, v, (row, col, ch))
    gauss1 = np.random.normal(0, v, (row, col, 3))
    noisy = image + gauss

    # Display original, Gaussian noise, and noisy images
    plt.figure(dpi=300)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    plt.subplot(131), plt.imshow(image), plt.title('Original')
    plt.subplot(132), plt.imshow(gauss1), plt.title('Gaussian(100%)')
    plt.subplot(133), plt.imshow(noisy), plt.title('Image with Gaussian(100%)')
    plt.savefig("NoiseAnalysis.png")
    plt.show()

    return noisy

def read_image(robotname):
    """
    Reads RGB and depth images from pickle files based on the robot type.

    Args:
    robotname (str): Name of the robot.
		-baxter (real-world environment only for DL-based HEC)
		-ur3 (real-world environment only for DL-based HEC)
		-ur3qrCL (real-world environment for CL-based HEC) 

    Returns:
    tuple: Lists of RGB images and depth images.
    """
    datar = list()
    datas = list()

    # Set total number of images and image size based on the robot type
    if robotname == 'baxter':
        totalN = 1615
        size = (455, 256)
    elif robotname == 'ur3':
        totalN = 2400
        size = (455, 256)
    elif robotname == 'ur3qr' or robotname == 'ur3qrCL':
        totalN = 2400
        size = (455, 256)
    else: # Sim environment for DL and CL-based HEC
        totalN = 5137
        size = (256, 256)

    for i in range(totalN):
        if robotname == 'ur3':
            pklr = open(str(mypath) + "/RDataset1/rgb_image/rgb_image%s.pkl" % (i + 1), 'rb')
        elif robotname == 'baxter':
            pklr = open(str(mypath) + "/RBDataset/rgb_image/rgb_image%s.pkl" % (i + 1), 'rb')
        elif robotname == 'ur3qr' or robotname == 'ur3qrCL':
            pklr = open(".../CalibrationwithImages_RWE_Blue_robot_roomCL/rgb_image/rgb_image%s.pkl" % (i + 1), 'rb')
        else:
            pklr = open(".../multipleRefs/rgb_image/rgb_image%s.pkl" % (i + 1), 'rb')

        imgr = pickle.load(pklr, encoding='latin1')
        imgr = cv2.resize(imgr, size).astype(np.float16)
        imgr = np.transpose(imgr, (2, 0, 1))

        if robotname == 'ur3':
            pkls = open(str(mypath) + "/Images/rgbd_image/rgbd_image%s.pkl" % (i + 1), 'rb')
        elif robotname == 'baxter':
            pkls = open(str(mypath) + "/Images/rgbd_image/rgbd_image%s.pkl" % (i + 1), 'rb')
        elif robotname == 'ur3qr' or robotname == 'ur3qrCL':
            pkls = open("/.../CalibrationwithImages_RWE_Blue_robot_roomCL/rgbd_image/rgbd_image%s.pkl" % (i + 1), 'rb')
        else:
            pkls = open("/Images/multipleRefs/rgbd_image/rgbd_image%s.pkl" % (i + 1), 'rb')

        imgs = pickle.load(pkls, encoding='latin1')
        imgs = cv2.resize(imgs, size)
        depth_max = np.nanmax(imgs)
        scale = 1.0 / depth_max
        depth_8bit = cv2.convertScaleAbs(imgs, None, scale)
        imgs = np.expand_dims(depth_8bit, 0)
        h = imgr
        h[3, :, :] = depth_8bit

        datar.append(imgr[:3, :, :])
        datas.append(imgs)

    return datar, datas

def read_pixel():
    """
    Reads target pixel data from a text file.

    Returns:
    list: List of target pixels.
    """
    data = list()
    with open(str(mypath) + "/Images/target_pixel.txt", 'r') as f:
        for line in f:
            linedata = line.split()
            temp = [float(j) for j in linedata]
            data.append(temp)
    return data

def read_input(robotname):
    """
    Reads input data (e.g., poses) from a text file based on the robot type.

    Args:
    robotname (str): Name of the robot.

    Returns:
    tuple: Lists of input data.
    """
    data = list()
    data1 = list()
    if robotname == 'ur3':
        myp = str(mypath) + "/RDataset1/input.txt"
    elif robotname == 'baxter':
        myp = str(mypath) + "/RBDataset/input.txt"
    elif robotname == 'ur3qr' or robotname == 'ur3qrCL':
        myp = ".../RWCL/CalibrationwithImages_RWE_Blue_robot_roomCL/input.txt"
    else:
        myp = ".../multipleRefs/base2refs20.txt"

    with open(myp, 'r') as f:
        for line in f:
            linedata = line.split()
            temp = [float(j) for j in linedata[:7]]
            temp1 = [float(j) for j in linedata[7:]]
            data.append(temp)
            data1.append(temp1)
    return data, data1

def camera_pose(robotname):
    """
    Reads camera pose data from a text file based on the robot type.

    Args:
    robotname (str): Name of the robot.

    Returns:
    list: List of camera poses.
    """
    data = list()
    if robotname == 'ur3':
        myp = str(mypath) + "/Images/camera_pose.txt"
    elif robotname == 'baxter':
        myp = str(mypath) + "/Iamegs/camera_pose.txt"
    elif robotname == 'ur3qr' or robotname == 'ur3qrCL':
        myp = ".../RWCL/CalibrationwithImages_RWE_Blue_robot_roomCL/camerapose.txt"
    else:
        myp = ".../multipleRefs/camera_pose.txt"

    with open(myp, 'r') as f:
        for line in f:
            linedata = line.split()
            temp = [float(j) for j in linedata]
            data.append(temp)
    return data

def cam2ref(robotname):
    """
    Reads camera to reference transformations from a text file based on the robot type.

    Args:
    robotname (str): Name of the robot.

    Returns:
    tuple: Lists of transformations.
    """
    data = list()
    data1 = list()
    if robotname == 'ur3':
        myp = str(mypath) + "/Images/input.txt"
    elif robotname == 'baxter':
        myp = str(mypath) + "/Images/input.txt"
    elif robotname == 'ur3qr' or robotname == 'ur3qrCL':
        myp = ".../RWCL/CalibrationwithImages_RWE_Blue_robot_roomCL/input.txt"
    else:
        myp = "../multipleRefs/camera2base.txt"

    with open(myp, 'r') as f:
        for line in f:
            linedata = line.split()
            temp = [float(j) for j in linedata[:7]]
            temp1 = [float(j) for j in linedata[7:]]
            data.append(temp)
            data1.append(temp1)
    return data, data1

def camera_pose1():
    """
    Returns a hardcoded list of camera poses for 1644 images.
    Likely used for testing or initialization purposes.

    Returns:
    list: List of camera poses.
    """
    data = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(1644)]
    return data
