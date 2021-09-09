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

torch.manual_seed(1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 14
# hidden_size = 500
hidden_size = 512
num_classes = 3
num_epochs = 50
batch_size = 50
learning_rate = 0.001

mypath = Path().absolute()

def read_data():
    base2ref = np.asarray(dt.read_input()[0])
    cam2ref = np.asarray(dt.read_input()[1])
    if num_classes==3:
        #base2ref = np.asarray(dt.read_input()[0])[:][:3]
        #cam2ref = np.asarray(dt.read_input()[1])[:][:3]
        label=np.asarray(dt.camera_pose())
        label=label[:,:3]
    else:
        #base2ref = np.asarray(dt.read_input()[0])[:][3:]
        #cam2ref = np.asarray(dt.read_input()[1])[:][3:]
        label=np.asarray(dt.camera_pose())
        label = label[:,3:]


    transformation=np.concatenate((base2ref,cam2ref),axis=1)
    transformation = torch.from_numpy(transformation)
    #cam2ref = torch.from_numpy(cam2ref)
    label = torch.from_numpy(label)
    Data_set = TensorDataset(transformation, label)
    return Data_set

def print_text(Resultt,loss_stats):
    #Resultt = open(str(mypath) + "/data5/" + "Design7.txt", "a")
    #for i in range(len(loss_stats['train'])):
    #    Resultt.write(str(loss_stats['train'][i])+"\t"+str(loss_stats['test'][i])+"\n")#str(loss_stats['train'][-1])+"\t"+str(loss_stats['val'][-1])+"\t"+str(loss_stats['test'][-1])+"\t")
    '''
    for i in loss_stats.values():
        for j in i:
            Resultt.write(str(j)+"\t")
        Resultt.write("\n")
        #Resultt.write(str(loss_stats['train'][i])+"\t"+str(loss_stats['test'][i])+"\n")#str(loss_stats['train'][-1])+"\t"+str(loss_stats['val'][-1])+"\t"+str(loss_stats['test'][-1])+"\t")
    '''
    for i in loss_stats:
        for j in i:
            Resultt.write(str(j) + "\t")
        Resultt.write("\n")

    #Resultt.write(str(xx)+"\t")
    #Resultt.write('\n')




# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc3 = nn.Linear(int(hidden_size / 2), int(hidden_size / 4))
        # self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(int(hidden_size / 4), num_classes)
        self.dropout1 = nn.Dropout(0.50)
        self.dropout2 = nn.Dropout(0.25)
        # self.fc5 = nn.Linear(int(hidden_size/8), num_classes)
        # self.fc6 = nn.Linear(8, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        # out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        # out = self.relu(out)
        # out = self.fc5(out)
        # out = self.relu(out)
        # out = self.fc6(out)

        return out







def Train(dataset1,n,b,l,h,ns):

    train_loader = torch.utils.data.DataLoader(dataset=dataset1['train'],
                                               batch_size=b,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset1['test'],
                                              batch_size=b,
                                              shuffle=False)

    model = NeuralNet(input_size, h, num_classes).to(device)

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = F.mse_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=l)
    # optimizer=torch.optim.SGD(model.parameters(), lr = learning_rate)
    # Train the model
    total_step = len(train_loader)

    temp_N = np.zeros((2, n))
    total_step = len(train_loader)
    for epoch in range(n):
        train_epoch_loss = 0
        for i, (transformation, labels) in enumerate(train_loader):
            # RGB = RGB.reshape(-1, 256 * 256*3).to(device)
            # RGBD = RGBD.reshape(-1, 256 * 256*1).to(device)
            # RGB = RGB.float()
            # RGBD = RGBD.float()

            if ns!=0:
                transformation=PS.noisyPose(transformation, ns,  0.017453292)

            #RGB = noisy(RGB, noise)
            transformation = transformation.to(device, dtype=torch.float)


            labels = labels.to(device, dtype=torch.float)
            outputs = model(transformation)
            # loss = criterion(outputs, pixel)
            # loss = criterion(outputs, labels)
            loss = criterion(outputs, labels)
            train_epoch_loss += loss.item()  # * 1000
            # train_epoch_loss += np.rad2deg(loss.item())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        # loss_stats['train'].append(train_epoch_loss / len(train_loader))
        temp_N[0][epoch] = (train_epoch_loss / len(train_loader))
        with torch.no_grad():
            test_loss = 0
            # total = 0
            # model.eval()
            for i, (transformation, labels) in enumerate(test_loader):
                # images = images.reshape(-1, 28*28).to(device)
                # labels = labels.to(device)
                if ns != 0:
                    transformation = PS.noisyPose(transformation, ns, 0.017453292)

                # RGB = noisy(RGB, noise)
                transformation = transformation.to(device, dtype=torch.float)

                labels = labels.to(device, dtype=torch.float)
                outputs = model(transformation)
                loss = criterion(outputs, labels)
                test_loss += loss.item()  # * 1000
                # train_epoch_loss += np.rad2deg(loss.item())
                # Backward and optimize

        # loss_stats['test'].append(test_loss / len(test_loader))
        temp_N[1][epoch] = (test_loss / len(test_loader))
    return temp_N




def train_val_dataset(dataset, val_split=0.20):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets



def run_experiment():
    dataset=read_data()
    y = train_val_dataset(dataset)
    #hidden_S = [64, 128, 256, 512]
    if num_classes==3:
        Resultt = open(str(mypath) + "/Result/RefnetResult/" + "RefnetTranslationNoiseApril.txt", "a")
    else:
        Resultt = open(str(mypath) + "/Result/RefnetResult/" + "RefnetOrientationNoiseApril.txt", "a")
    hidden_S = [512]
    num_epochs_L = [20]
    batch_size_L = [50]
    learning_rate_L = [0.001]
    repeat = 5
    noiseEnd=[0.0,0.001,0.003,0.005,0.008,0.01]
    #noiseEnd =[0.01]
    #noiseEnd=np.deg2rad(noiseEnd)
    for nindex, n in enumerate(num_epochs_L):
        for bindex, b in enumerate(batch_size_L):
            for lindex, l in enumerate(learning_rate_L):
                loss_stats_noise=list()

                for nsindex,ns in enumerate(noiseEnd):

                    #count += 1
                    loss_stats1 = {
                        'train': [],
                        "test": []
                    }

                    torch.manual_seed(1)
                    for rep in range(repeat):
                        f=Train(y,n,b,l,hidden_S[0],ns)
                        if rep ==0:
                            Cdict=f
                        else:
                            Cdict += f

                    loss_stats_noise.append(Cdict[0]/repeat)
                    loss_stats_noise.append(Cdict[1] / repeat)


                print_text(Resultt, loss_stats_noise)
            aaa=555




run_experiment()