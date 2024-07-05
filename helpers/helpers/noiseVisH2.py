import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

mypath = Path().absolute().parent
print(mypath)

def data_Noise():
    data = list()
    with open(str(mypath) + "/Result/OrientationResult_Unknown030421.txt",'r') as f:
    #with open(str(mypath) + "/Result/TranslationResult_H2.txt", 'r') as f:
        c=0
        for line in f:
            linedata = line.split()

            temp=list()
            #if c %2 ==1:
            for index,j in enumerate(linedata):
                temp.append(math.degrees(math.sqrt(float(j))))
                #temp.append(math.sqrt(float(j)*1000))
            data.append(temp)
            #c+=1
    return data

a=data_Noise()
a=np.asarray(a)
#noise = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.25]
# noise=[0]
#noiseEnd = [0, 1, 3, 5, 8, 10]
#n=[1,3,5,8,10]
a=a[:,:60]
n=[0, 1, 3, 5, 8, 10]
x=np.array([j for j in range(1,61)])

plt.figure(dpi=200)



for index,i in enumerate(a):
    if index==0:
        plt.plot(x, i, label="Train")
    elif index==1:
        plt.plot(x, i, label="Test")

plt.ylim(0, 30)

plt.legend(loc='upper right',fontsize=7)
# Add title and x, y labels
#plt.title("Translation Error for H2 Hypothesis", fontsize=10)
#plt.suptitle("Random Walk Suptitle", fontsize=10)
plt.xlabel("Epoch")
plt.ylabel("RMSE(radian)")

plt.savefig("H2Orientation1111.svg")
plt.show()