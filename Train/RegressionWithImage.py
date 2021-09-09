import dataset as dt
import numpy as np
import torch
from torch.utils.data import Subset
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from skimage.util import random_noise
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from helpers import PlotResult as PltR
from helpers import PoseNoise as PS


mypath = Path().absolute()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''
dataset_pix=dt.read_pixel()
dataset_pix=np.asarray(dataset_pix)[:][:]
dataset_im=dt.read_image()
dataset_tar=dt.read_input()[1]
#dataset_tar=dataset_tar[:3][:]#[:100][:]
dataset_tar=np.asarray(dataset_tar)
dataset_tar=dataset_tar[:,:]
dataset_r=dt.read_input()[0]
dataset_r=np.asarray(dataset_r)
dataset_r=dataset_r[:,:]
dataset_cam=np.asarray(dt.camera_pose())
dataset_cam=dataset_cam[:,3:]
data1= torch.from_numpy(np.asarray(dataset_im[0]))
data2=torch.from_numpy(np.asarray(dataset_im[1]))
data3=torch.from_numpy(dataset_r)
data4=torch.from_numpy(dataset_pix)
data5=torch.from_numpy(dataset_tar)
#temp6=np.asarray(dataset_im[2])
#data6=torch.from_numpy(temp6[:,3:])
data6=torch.from_numpy(dataset_cam)
Data_set=TensorDataset(data1,data2,data3,data4,data5,data6)
'''

input_size = 131080
input_size = 119081
input_size = 119079
#hidden_size = 500
hidden_size=512
num_classes = 4
hypoth=2
num_epochs = 50
batch_size = 50
learning_rate = 0.005

def read_data():

    knownTrans=np.asarray(dt.read_input()[0])
    dataset_im = dt.read_image()
    if hypoth==3:
        if num_classes==3:
            #base2ref = np.asarray(dt.read_input()[0])[:][:3]
            #cam2ref = np.asarray(dt.read_input()[1])[:][:3]
            label=np.asarray(dt.camera_pose())
            label=label[:,:3]
        elif num_classes==4:
            #base2ref = np.asarray(dt.read_input()[0])[:][3:]
            #cam2ref = np.asarray(dt.read_input()[1])[:][3:]
            label=np.asarray(dt.camera_pose())
            label = label[:,3:]
    else:
        if num_classes==3:
            label=np.asarray(dt.read_input()[1])[:,:3]
            #label2=2#label[:50][:3]
        else:
            label = np.asarray(dt.read_input()[1])[:,3:]
            #label = label[:50][3:]

    #transformation=np.concatenate((base2ref,cam2ref),axis=1)
    #transformation = torch.from_numpy(transformation)
    #cam2ref = torch.from_numpy(cam2ref)
    knownTrans=torch.from_numpy(knownTrans)
    label = torch.from_numpy(label)
    data1 = torch.from_numpy(np.asarray(dataset_im[0]))
    data2 = torch.from_numpy(np.asarray(dataset_im[1]))
    Data_set = TensorDataset(data1,data2,knownTrans, label)
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

def dual_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv


def noisy(image,ns):
    #mean = 0
    #var = ns
    #sigma = var ** 0.5
    #sigma = var
    #sigma=np.std(image)
    batch = image.view(image.size(0), image.size(1), -1)
    std=batch.std(2).sum(0)
    #std = torch.std(batch, dim=2)
    std1=std.cpu().detach().numpy()
    v = [(ns*i) ** 2 for i in std1]

    #gauss=[np.random.normal(0, v[i], (row, col)) for i in v]
    #gauss = np.random.normal(0, v, (row, col, ch))




    tens,ch,row,col= image.shape
    gauss = [np.random.normal(0, i, (tens,row, col)) for i in v]
    #gauss1=np.concatenate((gauss[0], gauss[1], gauss[2]), axis = 3)
    if ch==1:
        gauss1 = np.array([gauss[0]])
    else:
        gauss1=np.array([gauss[0], gauss[1], gauss[2]])
    #gauss = np.random.normal(0, v, (tens, ch, row, col))
    #gauss = np.random.normal(mean, sigma, (tens,ch,row, col))
    gauss = gauss1.reshape(tens,ch,row, col)

    noisy = image + gauss
    return noisy


# crop the image(tensor) to equal size
# as shown in architecture image , half left side image is concated with right side image
'''
def crop_tensor(target_tensor, tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]
'''

class Unet(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(Unet, self).__init__()

        # Left side (contracting path)
        #self.inp = input_conv(channels, 64)
        #self.dwn_conv1 = dual_conv(3, 64)
        #self.dwn_convD1=dual_conv(1, 64)o
        #self.dwn_conv2 = dual_conv(64, 128)
        #self.dwn_conv3 = dual_conv(128, 256)
        #self.dwn_conv4 = dual_conv(256, 512)
        #self.dwn_conv5 = dual_conv(512, 1024)
        self.dwn_conv1 = dual_conv(3, 8)
        self.dwn_convD1=dual_conv(1, 8)
        self.dwn_conv2 = dual_conv(8, 16)
        self.dwn_conv3 = dual_conv(16, 32)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Right side  (expnsion path)
        # transpose convolution is used showna as green arrow in architecture image
        self.trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = dual_conv(1024, 512)
        self.trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = dual_conv(512, 256)
        self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = dual_conv(256, 128)
        self.trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = dual_conv(128, 64)

        # output layer
        self.out = nn.Conv2d(64, 2, kernel_size=1)
        #FFNN
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc3 = nn.Linear(int(hidden_size / 2), int(hidden_size / 4))
        # self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(int(hidden_size / 4), num_classes)
        # self.dropout1 = nn.Dropout(0.50)
        # self.dropout2 = nn.Dropout(0.25)
        # self.fc5 = nn.Linear(int(hidden_size/8), num_classes)
        # self.fc6 = nn.Linear(8, num_classes)

    def forward(self, RGB,RGBD,kTrans):
        # forward pass for Left side RGBo
        x1 = self.dwn_conv1(RGB)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        #x5 = self.dwn_conv3(x4)
        #x6 = self.maxpool(x5)
        #x7 = self.dwn_conv4(x6)
        #x8 = self.maxpool(x7)
        #x9 = self.dwn_conv5(x8)

        # forward pass for Left side RGBD
        xx1 = self.dwn_convD1(RGBD)
        xx2 = self.maxpool(xx1)
        xx3 = self.dwn_conv2(xx2)
        xx4 = self.maxpool(xx3)
        #xx5 = self.dwn_conv3(xx4)o
        #xx6 = self.maxpool(xx5)
        #xx7 = self.dwn_conv4(xx6)
        #xx8 = self.maxpool(xx7)
        #xx9 = self.dwn_conv5(xx8)

        # forward pass for concatenated values

        y1=kTrans
        #y2=pixel
        #x11=torch.cat((y1,y2), dim=1)
        #x10 = torch.cat((x9, xx9), dim=1)
        f1=torch.flatten(x4,start_dim=1)
        f2=torch.flatten(xx4,start_dim=1)
        x10 = torch.cat((f1, f2), dim=1)
        #x12=torch.cat((x10,x11), dim=1)
        x12 = torch.cat((x10, y1), dim=1)
        x13=torch.flatten(x12,start_dim=1)
        out = self.fc1(x13)
        out = self.relu(out)
        # out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        # out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        '''
        # forward pass for Right side
        x = self.trans1(x9)
        y = crop_tensor(x, x7)
        x = self.up_conv1(torch.cat([x, y], 1))

        x = self.trans2(x)
        y = crop_tensor(x, x5)
        x = self.up_conv2(torch.cat([x, y], 1))

        x = self.trans3(x)
        y = crop_tensor(x, x3)
        x = self.up_conv3(torch.cat([x, y], 1))

        x = self.trans4(x)
        y = crop_tensor(x, x1)
        x = self.up_conv4(torch.cat([x, y], 1))

        x = self.out(x)
        '''
        return out

def train_val_dataset(dataset, val_split=0.20):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets

def Train(dataset1,n,b1,l,noise,ns):
    #count=0
    #Resultt = open(str(mypath) + "/Result/" + "result9.txt", "a")
    '''
    hidden_S = [64, 128, 256, 512]
    num_epochs_L = [50, 100, 200, 300]
    batch_size_L = [50, 100, 250, 500]
    learning_rate_L = [0.001, 0.005, 0.01]
    repeat=5
    '''
    hidden_S = 512
    num_epochs_L = n
    b = b1
    learning_rate_L = l
    #repeat=1

    #Resultt = open(str(mypath) + "Design1.txt", "a")
    #for n in range(num_epochs_L):
    #for b in batch_size_L:

    train_loader = torch.utils.data.DataLoader(dataset=dataset1['train'],
                                               batch_size=b,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset1['test'],
                                             batch_size=b,
                                             shuffle=False)
    model = Unet(input_size,hidden_S, num_classes).to(device)


    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = F.mse_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_L)
    #loss_stats = {
    #    'train': [],
    #    "test": []
    #}
    temp_N = np.zeros((2, num_epochs_L))
    total_step = len(train_loader)
    for epoch in range(num_epochs_L):
        train_epoch_loss = 0
        #for i, (RGB,RGBD,kTrans,pixel,labels,camera) in enumerate(train_loader):
        for i, (RGB, RGBD, kTrans,labels) in enumerate(train_loader):
            #RGB = RGB.reshape(-1, 256 * 256*3).to(device)
            #RGBD = RGBD.reshape(-1, 256 * 256*1).to(device)
            #RGB = RGB.float()
            #RGBD = RGBD.float()
            if noise!=0:
                RGB=noisy(RGB,noise)
                RGBD = noisy(RGBD, noise)
            RGB = RGB.to(device,dtype=torch.float)
            #RGBD = noisy(RGBD,noise)
            RGBD = RGBD.to(device,dtype=torch.float)

            #kTrans=PS.noisyPose(kTrans,ns,0.05)
            if ns!=0:
                kTrans = PS.noisyPose(kTrans, ns, 0.017453292)
            kTrans = kTrans.to(device, dtype=torch.float)
            #pixel=pixel.to(device,dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            #camera = camera.to(device, dtype=torch.float)
            outputs = model(RGB,RGBD,kTrans)
            #loss = criterion(outputs, pixel)
            #loss = criterion(outputs, labels)
            #loss = criterion(outputs, labels)
            loss=torch.sqrt(criterion(outputs, labels))
            train_epoch_loss += loss.item() #* 1000
            # train_epoch_loss += np.rad2deg(loss.item())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        #loss_stats['train'].append(train_epoch_loss / len(train_loader))
        temp_N[0][epoch] = (train_epoch_loss / len(train_loader))
        with torch.no_grad():
            test_loss = 0
            # total = 0
            # model.eval()
            #for i, (RGB,RGBD,kTrans,pixel,labels,camera) in enumerate(test_loader):
            for i, (RGB, RGBD, kTrans,labels) in enumerate(test_loader):
                # images = images.reshape(-1, 28*28).to(device)
                # labels = labels.to(device)
                RGB = RGB.to(device, dtype=torch.float)
                RGBD = RGBD.to(device, dtype=torch.float)
                kTrans = kTrans.to(device, dtype=torch.float)
                #pixel = pixel.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                #camera = camera.to(device, dtype=torch.float)
                outputs = model(RGB, RGBD, kTrans)
                #loss = criterion(outputs, labels)
                #loss = criterion(outputs, pixel)
                #loss = criterion(outputs, labels)
                loss = torch.sqrt(criterion(outputs, labels))
                test_loss += loss.item()  # * 1000
                # train_epoch_loss += np.rad2deg(loss.item())
                # Backward and optimize

        #loss_stats['test'].append(test_loss / len(test_loader))
        temp_N[1][epoch] = (test_loss / len(test_loader))


    #print_text(Resultt,loss_stats)

    return temp_N





def run_experiment():
    y = train_val_dataset(read_data())
    #hidden_S = [64, 128, 256, 512]
    if hypoth==3:
        if num_classes==3:
            Resultt = open(str(mypath) + "/Result/" + "H3Translation.txt", "a")
        elif num_classes==4:
            Resultt= open(str(mypath) + "/Result/" + "H3Orientation.txt", "a")
    else:
        if num_classes==3:
            Resultt = open(str(mypath) + "/Result/" + "H2TranslationWithNoise.txt", "a")
        else:
            Resultt = open(str(mypath) + "/Result/" + "H2OrientationWithNoise.txt", "a")
    num_epochs_L = [50, 100, 200]
    batch_size_L = [4, 8, 16, 32]
    learning_rate_L = [0.001, 0.005]
    num_epochs_L = [60]
    batch_size_L = [32]
    learning_rate_L = [0.01]
    repeat = 5
    count=600
    noise=[0.0,0.01,0.02,0.05,0.08,0.1,0.25]
    #noise=[0.0]
    noiseEnd=[0.0,0.001,0.003,0.005,0.008,0.01]
    #noiseEnd=[0.0]#np.deg2rad(noiseEnd)
    for nindex, n in enumerate(num_epochs_L):
        for bindex, b in enumerate(batch_size_L):
            for lindex, l in enumerate(learning_rate_L):

                loss_stats_noise=list()
                for imnsindex,imns in enumerate(noise):
                    for nsindex,ns in enumerate(noiseEnd):

                        count += 1
                        loss_stats1 = {
                            'train': [],
                            "test": []
                        }

                        torch.manual_seed(1)
                        for rep in range(repeat):
                            f=Train(y,n,b,l,imns,ns)
                            if rep ==0:
                                Cdict=f
                            else:
                                Cdict += f

                        loss_stats_noise.append(Cdict[0]/repeat)
                        loss_stats_noise.append(Cdict[1] / repeat)

                        #[loss_stats1['train'].append((t) / repeat) for t in Cdict[0]]
                        #[loss_stats1['test'].append((t) / repeat) for t in Cdict[1]]

                        #loss_stats_noise['train'].append((Cdict[0, -1] ) / repeat)
                        #[loss_stats1['test'].append((t) / repeat) for t in Cdict[1]]
                        #loss_stats_noise['test'].append((Cdict[1, -1] ) / repeat)
                        #print_text(Resultt2,loss_stats1) #print data without noise
                        #plot(loss_stats1, count, n, b, l) plot data without noise
                #loss_stats_noise['train'].append(loss_stats1['train'][-1])
                #loss_stats_noise['test'].append(loss_stats1['test'][-1])
                #PltR.plot_noise(loss_stats_noise, count, n, b, l)
                print_text(Resultt, loss_stats_noise)
            aaa=555








run_experiment()

#y=train_val_dataset(Data_set)
#f=Train(y)
#a=5