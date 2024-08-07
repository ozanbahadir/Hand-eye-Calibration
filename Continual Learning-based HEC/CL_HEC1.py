from torch.utils.data import Subset, TensorDataset, DataLoader
from pathlib import Path
import torch.nn.functional as F
from helpers import PoseNoise as PS
from helpers import print_data
from HECmodels import RotationNet, UnetR, noisy
import torch
from quatLosses import *

# Set device and paths
mypath = Path().absolute()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 119079
hidden_size = 512
num_epochs = 50
batch_size = 50
learning_rate = 0.005
robotname = 'CL'

# Evaluation function
def evaluation(model, train_loaderT, criterion):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        test_loss_ev = 0
        cancel = False

        for ii, (RGB, RGBD, kTrans, labels) in enumerate(train_loaderT):
            RGB = RGB.to(device, dtype=torch.float)
            if RGB.shape[0] != 100:
                RGBD = RGBD.to(device, dtype=torch.float)
                kTrans = kTrans.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(RGB, RGBD, kTrans)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                if num_classes == 3:
                    lossev1 = math.sqrt(loss * 1000)
                    test_loss_ev += lossev1
                else:
                    test_loss_ev += quat_angle_diff(outputs, labels)

        test_loss_ev = (test_loss_ev / len(train_loaderT))
        if num_classes == 3:
            if test_loss_ev <= 3:
                cancel = True
        else:
            if test_loss_ev <= 4:
                cancel = True
        return cancel

# Training function
def Train(y, Resultt, n, b1, l, noise, ns, rep, buf, spe):
    hidden_S = 512
    num_epochs_L = n
    b = b1
    learning_rate_L = l
    cn = False

    if robotname == 'CL':
        n_subset = 71
        t_subset = 6
    else:
        n_subset = 14
        t_subset = 3

    idxlist = [False for ff in range(n_subset)]
    for s in range(n_subset):
        dataset1 = print_data.train_val_datasetCN(y, robotname, s, buf, idxlist)
        if spe == True and s != 0:
            train_loader = torch.utils.data.DataLoader(dataset=dataset1['train'], batch_size=b, shuffle=True)
            cn = evaluation(model, train_loader, criterion)
            idxlist[s] = cn
            if buf:
                train_dev_sets = torch.utils.data.ConcatDataset([dataset1['train'], dataset1['buffer']])
                train_loader = DataLoader(dataset=train_dev_sets, batch_size=b, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=dataset1['train'], batch_size=b, shuffle=True)

        if s == 0:
            if num_classes == 3:# Estimate the Translation component of the HEC
                model = UnetR(input_size, hidden_S, num_classes).to(device)
                criterion = F.mse_loss
            else:#Estimate the Orientation component of the HEC
                model = RotationNet(input_size, hidden_S, num_classes).to(device)
                criterion = quat_chordal_squared_loss
        else:
            model = model.to(device)

        if cn == False:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_L)
            temp_N = np.zeros((n_subset + 1, num_epochs_L))
            total_step = len(train_loader)
            for epoch in range(num_epochs_L):
                train_epoch_loss = 0
                train_epoch_loss_ev = 0
                for i, (RGB, RGBD, kTrans, labels) in enumerate(train_loader):
                    if noise != 0:
                        RGB = noisy(RGB, noise)
                        RGBD = noisy(RGBD, noise)

                    RGB = RGB.to(device, dtype=torch.float)
                    if RGB.shape[0] != 100:
                        RGBD = RGBD.to(device, dtype=torch.float)
                        if ns != 0:
                            kTrans = PS.noisyPose(kTrans, ns, 0.017453292)
                        kTrans = kTrans.to(device, dtype=torch.float)
                        labels = labels.to(device, dtype=torch.float)
                        outputs = model(RGB, RGBD, kTrans)
                        loss = criterion(outputs, labels)
                        train_epoch_loss += loss.item()
                        train_epoch_loss_ev += quat_angle_diff(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if (i + 1) % 1 == 0:
                            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

                temp_N[0][epoch] = (train_epoch_loss_ev / len(train_loader))
        else:
            temp_N[0][epoch] = -100

        for l in range(t_subset):
            test_loader_name = 'test%s' % (l + 1)
            test_loader = torch.utils.data.DataLoader(dataset=dataset1[test_loader_name], batch_size=b, shuffle=False)
            with torch.no_grad():
                model.eval()
                test_loss_ev = 0
                for ii, (RGB, RGBD, kTrans, labels) in enumerate(test_loader):
                    RGB = RGB.to(device, dtype=torch.float)
                    if RGB.shape[0] != 100:
                        RGBD = RGBD.to(device, dtype=torch.float)
                        kTrans = kTrans.to(device, dtype=torch.float)
                        labels = labels.to(device, dtype=torch.float)
                        outputs = model(RGB, RGBD, kTrans)
                        loss = criterion(outputs, labels)
                        test_loss_ev += quat_angle_diff(outputs, labels)
                temp_N[l + 1][epoch] = (test_loss_ev / len(test_loader))

        print_data.print_text(Resultt, temp_N)

    return temp_N

# Function to run the experiment
def run_experiment(y, buf, spe):
    if hypoth == 3:
        if num_classes == 3:
            Resultt = open(str(mypath) + "/Result/CL/Sim/" + "H3TranslationRandomBufferCLRW115.txt", "a")
        else:
            Resultt = open(str(mypath) + "/Result/CL/Sim/" + "H3OrientationRandomBufferCLRW115.txt", "a")
    else:
        if num_classes == 3:
            Resultt = open(str(mypath) + "/Result/RefnetResult/" + "H2TranslationSimNovember2022.txt", "a")
        else:
            Resultt = open(str(mypath) + "/Result/RefnetResult/" + "H2OrientationSimNovember2022.txt", "a")

    num_epochs_L = [30]
    batch_size_L = [4]
    if num_classes == 3:
        num_epochs_L = [20]
        learning_rate_L = [0.0001]
    else:
        learning_rate_L = [0.00001]

    repeat = 3
    count = 600
    noise = [0.0]
    noiseEnd = [0.0]

    for nindex, n in enumerate(num_epochs_L):
        for bindex, b in enumerate(batch_size_L):
            for lindex, l in enumerate(learning_rate_L):
                for imnsindex, imns in enumerate(noise):
                    for nsindex, ns in enumerate(noiseEnd):
                        count += 1
                        torch.manual_seed(1)
                        for rep in range(repeat):
                            torch.manual_seed(1)
                            f = Train(y, Resultt, n, b, l, imns, ns, rep, buf, spe)
                            if rep == 0:
                                Cdict = f
                            else:
                                Cdict += f

num_classes1 = [10]
hypoth1 = [3]
im = True
buf = True #Activate the CL 
spe = True

img, knw, cm, cm2rf = print_data.read_data(robotname, im)
for i in hypoth1:
    for j in num_classes1:
        hypoth = i
        num_classes = j
        merged_data = print_data.merge_data(img, knw, cm, hypoth, num_classes, cm2rf)
        run_experiment(merged_data, buf, spe)
