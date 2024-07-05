from torch.utils.data import Subset
from pathlib import Path
import torch.nn.functional as F
from helpers import PoseNoise as PS
from helpers import print_data
from HECmodels import *

# Get the current absolute path
mypath = Path().absolute()
# Check if CUDA is available, set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters and configurations
input_size = 131080
input_size = 119081
input_size = 119079
input_size = 94983
input_size = 36871
input_size = 10247
input_size = 119079
input_size = 32775
input_size = 81927
hidden_size = 512
num_epochs = 50
batch_size = 50
learning_rate = 0.005
robotname = 'Sim'
robotname = 'ur3qr'

# Training function
def Train(dataset1, n, b1, l, noise, ns, rep):
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
	
	hidden_S = 512
    num_epochs_L = n
    b = b1
    learning_rate_L = l

    # Initialize the model based on the number of classes
    if num_classes == 3:
        criterion = F.mse_loss
        model = UnetR(input_size, hidden_S, num_classes).to(device)
    else:
        model = RotationNet(input_size, hidden_S, num_classes).to(device)
        criterion = quat_chordal_squared_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_L)

    temp_N = np.zeros((4, num_epochs_L))
    total_step = len(train_loader)
    
    # Training loop
    for epoch in range(num_epochs_L):
        train_epoch_loss = 0
        train_epoch_loss_ev = 0
        
        for i, (RGB, RGBD, kTrans, labels) in enumerate(train_loader):
            if noise != 0:
                RGB = noisy(RGB, noise)
                RGBD = noisy(RGBD, noise)
            
            RGB = RGB.to(device, dtype=torch.float)
            RGBD = RGBD.to(device, dtype=torch.float)
            
            if ns != 0:
                kTrans = PS.noisyPose(kTrans, ns, 0.017453292)
            
            kTrans = kTrans.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            
            outputs = model(RGB, RGBD, kTrans)
            
            if num_classes == 3:
                loss = criterion(outputs, labels)
                train_epoch_loss += loss.item()

                lossev = math.sqrt(loss * 1000)
                train_epoch_loss_ev += lossev

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

            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
        temp_N[0][epoch] = (train_epoch_loss / len(train_loader))
        temp_N[1][epoch] = (train_epoch_loss_ev / len(train_loader))
        
        # Evaluation on the test set
        with torch.no_grad():
            test_loss = 0
            test_loss_ev = 0

            model.eval()
            for i, (RGB, RGBD, kTrans, labels) in enumerate(test_loader):
                RGB = RGB.to(device, dtype=torch.float)
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
        
        temp_N[2][epoch] = (test_loss / len(test_loader))
        temp_N[3][epoch] = (test_loss_ev / len(test_loader))
    
    # Save the model
    if hypoth == 3:
        if num_classes == 3:
            pp = str(mypath) + '/models/SimH3/translation/modelNovember2022%s' % (rep + 1)
        else:
            pp = str(mypath) + '/models/SimH3/orientation/modelNovember2022%s' % (rep + 1)
    else:
        if num_classes == 3:
            pp = str(mypath) + '/models/SimH2/translation/model%s' % (rep + 1)
        else:
            pp = str(mypath) + '/models/SimH2/orientation/model%s' % (rep + 1)
    
    torch.save(model, pp)
    
    return temp_N

# Experiment runner function
def run_experiment(mdata, buf):
    y = print_data.train_val_dataset(mdata, robotname, 0, buf)
    
    if hypoth == 3:
        if num_classes == 3:
            Resultt = open(str(mypath) + "/Result/CL/Sim/" + "H3TranslationRandom100.txt", "a")
        else:
            Resultt = open(str(mypath) + "/Result/CL/Sim/" + "H3OrientationRandom100.txt", "a")
    else:
        if num_classes == 3:
            Resultt = open(str(mypath) + "/Result/RefnetResult/" + "H2TranslationSimNovember2022.txt", "a")
        else:
            Resultt = open(str(mypath) + "/Result/RefnetResult/" + "H2OrientationSimNovember2022.txt", "a")
    
    num_epochs_L = [15, 30, 50, 100]
    batch_size_L = [4, 8, 16, 32]
    learning_rate_L = [0.001, 0.005, 0.01]
    num_epochs_L = [30]
    batch_size_L = [4]
    learning_rate_L = [0.000001]
    
    if num_classes == 3:
        learning_rate_L = [0.0001]
        num_epochs_L = [10]
    else:
        learning_rate_L = [0.00001]
        num_epochs_L = [20]
    
    repeat = 3
    count = 600
    noise = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.25]
    noise = [0.0]
    noiseEnd = [0.0, 0.001, 0.003, 0.005, 0.008, 0.01]
    noiseEnd = [0.0]
    
    for nindex, n in enumerate(num_epochs_L):
        for bindex, b in enumerate(batch_size_L):
            for lindex, l in enumerate(learning_rate_L):
                
                loss_stats_noise = list()
                for imnsindex, imns in enumerate(noise):
                    for nsindex, ns in enumerate(noiseEnd):
                        count += 1
                        loss_stats1 = {'train': [], "test": []}

                        torch.manual_seed(1)
                        for rep in range(repeat):
                            f = Train(y, n, b, l, imns, ns, rep)
                            if rep == 0:
                                Cdict = f
                            else:
                                Cdict += f
                                print_data.print_text(Resultt, f)

                aaa = 555

num_classes1 = [3,10]# 3 for Translation and 10 for Orientation component of the HEC
hypoth1 = [2,3] # check the paper for the explanation of hypotheses
im = True
buf = False # Used for CL not for DL-based HEC
img, knw, cm, cm2rf = print_data.read_data(robotname, im)

for i in hypoth1:
    for j in num_classes1:
        hypoth = i
        num_classes = j
        merged_data = print_data.merge_data(img, knw, cm, hypoth, num_classes, cm2rf)
        run_experiment(merged_data, buf)
