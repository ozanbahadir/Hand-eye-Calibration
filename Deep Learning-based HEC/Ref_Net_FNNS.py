import torch
import torch.nn as nn
import dataset as dt
import torchvision
import torchvision.transforms as transforms
from skimage.util import random_noise
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
from mpl_toolkits import mplot3d
from helpers import PoseNoise as PS
from mathutils import Quaternion
from quat2A import *
from quatLosses import *
import time

from  HECmodels import *

# Set seed for reproducibility
torch.manual_seed(1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 14  # Define the input size for your neural network
hidden_size = 1024  # Define the size of the hidden layer(s)
num_classes = 7  # Define the number of output classes or dimensions
num_epochs = 50  # Number of times to iterate over the dataset during training
batch_size = 50  # Number of samples to work through before updating the model
learning_rate = 0.001  # Learning rate for the optimizer

mypath = Path().absolute()
robotname='null'

def Train(dataset1, n, b, l, h, ns, rep):
    """
    Function to train the neural network model.

    Args:
    - dataset1 (dict): Dictionary containing 'train' and 'test' datasets
    - n (int): Number of epochs to train the model
    - b (int): Batch size for training and testing
    - l (float): Learning rate for the optimizer
    - h (int): Size of the hidden layer in the neural network
    - ns (float): Noise level for pose data augmentation
    - rep (int): Repetition number for saving models

    Returns:
    - temp_N (numpy.ndarray): Array containing training and testing loss values
    """

    # Data loaders for training and testing datasets
    train_loader = torch.utils.data.DataLoader(dataset=dataset1['train'],
                                               batch_size=b,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset1['test'],
                                              batch_size=b,
                                              shuffle=False)

    # Define the model architecture based on the number of output classes
    if num_classes == 3:
        model = FNNNet(input_size, h, num_classes).to(device)
        criterion = F.mse_loss  # Mean Squared Error loss for regression task
    else:
        model = FNNNet(input_size, h, num_classes).to(device)
        criterion = quat_chordal_squared_loss  # Custom loss for quaternion regression

    optimizer = torch.optim.Adam(model.parameters(), lr=l)  # Adam optimizer

    total_step = len(train_loader)  # Total number of batches in the training dataset

    temp_N = np.zeros((4, n))  # Array to store loss values across epochs

    # Training loop
    for epoch in range(n):
        train_epoch_loss = 0
        train_epoch_loss_ev = 0

        # Iterate over batches in the training dataset
        for i, (transformation, labels) in enumerate(train_loader):
            transformation = transformation.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            # Forward pass
            outputs = model(transformation)

            # Compute loss
            if num_classes == 3:
                loss = criterion(outputs, labels)
                train_epoch_loss += loss.item()

                lossev = math.sqrt(loss * 1000)
                train_epoch_loss_ev += lossev

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss = criterion(outputs, labels)
                train_epoch_loss += loss.item()
                train_epoch_loss_ev += quat_angle_diff(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print training loss every 10 steps
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Store average training loss for the epoch
        temp_N[0][epoch] = (train_epoch_loss / len(train_loader))
        temp_N[1][epoch] = (train_epoch_loss_ev / len(train_loader))

        # Evaluate on the test dataset
        with torch.no_grad():
            test_loss = 0
            test_loss_ev = 0

            # Iterate over batches in the test dataset
            for i, (transformation, labels) in enumerate(test_loader):
                if ns != 0:
                    transformation = PS.noisyPose(transformation, ns, 0.017453292)

                transformation = transformation.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                # Forward pass
                outputs = model(transformation)

                # Compute test loss
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                if num_classes == 3:
                    lossev1 = math.sqrt(loss * 1000)
                    test_loss_ev += lossev1
                else:
                    test_loss_ev += quat_angle_diff(outputs, labels)

        # Store average test loss for the epoch
        temp_N[2][epoch] = (test_loss / len(test_loader))
        temp_N[3][epoch] = (test_loss_ev / len(test_loader))

    # Save the trained model
    if num_classes == 3:
        pp = str(mypath) + '/models/refnet/translation/modelN%s' % (rep + 1)
    else:
        pp = str(mypath) + '/models/refnet/orientation/modelN33%s' % (rep + 1)
    torch.save(model, pp)

    return temp_N


def run_experiment(y):
    """
    Function to run multiple experiments varying hyperparameters.

    Args:
    - y (dict): Merged dataset dictionary containing 'train' and 'test' datasets
    """

    dataset1 = print_data.train_val_dataset(y, robotname, False)

    # Open file to log results
    if num_classes == 3:
        Resultt = open(str(mypath) + "/Result/RefnetResult/" + "RefnetTranslation2022N2.txt", "a")
    else:
        Resultt = open(str(mypath) + "/Result/RefnetResult/" + "RefNetFusion.txt", "a")

    # Define lists of hyperparameters to iterate over
    hidden_S = [2048]
    num_epochs_L = [50, 100]  # Number of epochs to try
    batch_size_L = [50]  # Batch sizes to try
    learning_rate_L = [0.0001, 0.001]  # Learning rates to try
    repeat = 3  # Number of repetitions for each configuration

    noiseEnd = [0]  # Noise levels to try

    # Loop over hyperparameter combinations
    for nindex, n in enumerate(num_epochs_L):
        for bindex, b in enumerate(batch_size_L):
            for lindex, l in enumerate(learning_rate_L):
                loss_stats_noise = list()

                for nsindex, ns in enumerate(noiseEnd):
                    loss_stats1 = {
                        'train': [],
                        "test": []
                    }

                    # Train the model multiple times for averaging results
                    for rep in range(repeat):
                        f = Train(dataset1, n, b, l, hidden_S[0], ns, rep)
                        print_data.print_text(Resultt, f)

          

# Set the values of num_classes and hypoth1 for execution
num_classes = [3]
hypoth1 = [1]
im = False

# Read data based on robotname and hypothetical scenario
img, knw, cm, cm2rf = print_data.read_data(robotname, im)

# Loop over hypothetical scenarios and number of classes
for i in hypoth1:
    for j in num_classes:
        hypoth = i
        num_classes = j
        merged_data = print_data.merge_data(img, knw, cm, hypoth, num_classes, cm2rf)
        run_experiment(merged_data)
