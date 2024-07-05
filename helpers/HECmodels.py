import torch.nn as nn
from mathutils import Quaternion
from helpers import quat2A 
from helpers import quatLosses





class RotationNet(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(RotationNet, self).__init__()
        self.net = UnetR(input_size, hidden_size, num_classes)  # Outputs Bx10

    def A_vec_to_quat(self, A_vec):
        A = convert_Avec_to_A(A_vec)  # Bx10 -> Bx4x4
        sizeR = A.size()
        lsizeR = len(sizeR)
        if  lsizeR!=3:
            A = torch.unsqueeze(A, dim=0)
        _, evs = torch.symeig(A, eigenvectors=True)

        #if lsizeR!=3:
            #return evs[:, 0].squeeze()

       # else:
        return evs[:, :, 0].squeeze()

    def forward(self,RGB,RGBD,kTrans):
        A_vec = self.net( RGB,RGBD,kTrans)  # Bx10
        q = self.A_vec_to_quat(A_vec)  # Bx10 -> Bx4
        sizeq=q.size()
        if len(sizeq)!=2:
            q = torch.unsqueeze(q, dim=0)
        return q  # unit quaternions!

def quaternion_loss(outputs,labels):
    ornNd = outputs.detach().cpu().numpy()
    ornQuat = [Quaternion(i).normalized() for i in ornNd]
    ls=0
    for indexi,i in enumerate(ornQuat):
		ls+=1-math.pow(Quaternion.dot(i,labels[indexi]),2)
    lss=torch.tensor(np.array(ls))
    return lss







def dual_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv


def noisy(image,ns):

    batch = image.view(image.size(0), image.size(1), -1)
    std=batch.std(2).sum(0)
    std1=std.cpu().detach().numpy()
    v = [(ns*i) ** 2 for i in std1]





    tens,ch,row,col= image.shape
    gauss = [np.random.normal(0, i, (tens,row, col)) for i in v]

    if ch==1:
        gauss1 = np.array([gauss[0]])
    else:
        gauss1=np.array([gauss[0], gauss[1], gauss[2]])

    gauss = gauss1.reshape(tens,ch,row, col)

    noisy = image + gauss
    return noisy




class UnetR(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(UnetR, self).__init__()

        self.dwn_conv1 = dual_conv(3, 64)
        self.dwn_convD1=dual_conv(1, 64)
        self.dwn_conv2 = dual_conv(64,128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512,1024)
        self.dwn_conv6 = dual_conv(1024,2048)
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
  
        self.fc4 = nn.Linear(int(hidden_size / 4), num_classes)
  

    def forward(self, RGB,RGBD,kTrans):
        # forward pass for Left side RGBo
        x1 = self.dwn_conv1(RGB)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)
        x90 = self.maxpool(x9)
  
        xx1 = self.dwn_convD1(RGBD)
        xx2 = self.maxpool(xx1)
        xx3 = self.dwn_conv2(xx2)
        xx4 = self.maxpool(xx3)
        xx5 = self.dwn_conv3(xx4)
        xx6 = self.maxpool(xx5)
        xx7 = self.dwn_conv4(xx6)
        xx8 = self.maxpool(xx7)
        xx9 = self.dwn_conv5(xx8)
        xx90 = self.maxpool(xx9)
   

        # forward pass for concatenated values

        y1=kTrans
      
        f1=torch.flatten(x90,start_dim=1)
        f2=torch.flatten(xx90,start_dim=1)
        x10 = torch.cat((f1, f2), dim=1)
     
        x12 = torch.cat((x10, y1), dim=1)
        x13=torch.flatten(x12,start_dim=1)
        out = self.fc1(x13)
        out = self.relu(out)
  
        out = self.fc2(out)
        out = self.relu(out)
       
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
  
        return out





class UnetS(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(UnetS, self).__init__()
        self.dwn_conv1 = dual_conv(3, 8)
        self.dwn_convD1=dual_conv(1, 8)
        self.dwn_conv2 = dual_conv(8, 16)
        self.dwn_conv3 = dual_conv(16, 32)
        self.dwn_conv4 = dual_conv(32, 64)
        self.dwn_conv5 = dual_conv(64,128)
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

   
        self.out = nn.Conv2d(64, 2, kernel_size=1)
        #FFNN


        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc3 = nn.Linear(int(hidden_size / 2), int(hidden_size / 4))
        self.fc4 = nn.Linear(int(hidden_size / 4), num_classes)
      

    def forward(self, RGB,RGBD,kTrans):
        # forward pass for Left side RGBo
        x1 = self.dwn_conv1(RGB)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)
        x90 = self.maxpool(x9)
        # forward pass for Left side RGBD
        xx1 = self.dwn_convD1(RGBD)
        xx2 = self.maxpool(xx1)
        xx3 = self.dwn_conv2(xx2)
        xx4 = self.maxpool(xx3)
        xx5 = self.dwn_conv3(xx4)
        xx6 = self.maxpool(xx5)
        xx7 = self.dwn_conv4(xx6)
        xx8 = self.maxpool(xx7)
        xx9 = self.dwn_conv5(xx8)
        xx90 = self.maxpool(xx9)

        # forward pass for concatenated values

        y1=kTrans
       
        f1=torch.flatten(x4,start_dim=1)
        f2=torch.flatten(xx4,start_dim=1)
        x10 = torch.cat((f1, f2), dim=1)
      
        x12 = torch.cat((x10, y1), dim=1)
        x13=torch.flatten(x12,start_dim=1)
        out = self.fc1(x13)
        out = self.relu(out)
    
        out = self.fc2(out)
        out = self.relu(out)
   
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)

        return out

# Fully connected neural network with one hidden layer
class FNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FNNNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc3 = nn.Linear(int(hidden_size / 2), int(hidden_size / 4))
        # self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(int(hidden_size / 4), num_classes)
        self.dropout1 = nn.Dropout(0.50)
        self.dropout2 = nn.Dropout(0.25)
        self.fc5 = nn.Linear(int(hidden_size/8), int(hidden_size / 16))
        self.fc6 = nn.Linear(int(hidden_size / 16), num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        #out = self.relu(out)
        #out = self.fc5(out)
        #out = self.relu(out)
        #out = self.fc6(out)

        return out
