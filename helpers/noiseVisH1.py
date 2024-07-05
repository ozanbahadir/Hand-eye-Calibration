import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
from matplotlib.ticker import MaxNLocator

mypath = Path().absolute().parent
print(mypath)

def data_Noise():
    data = list()
    with open(str(mypath) + "/Result/RefnetResult/RefnetOrientationNoiseApril.txt",'r') as f:
    #with open(str(mypath) + "/Result/RefnetResult/RefnetTranslationNoiseApril.txt", 'r') as f:
        c=0
        for line in f:
            linedata = line.split()

            temp=list()
            if c %2 ==1:
                for index,j in enumerate(linedata):
                    temp.append(math.degrees(math.sqrt(float(j))))
                    #temp.append(math.sqrt(float(j)*1000))
                data.append(temp)
            c+=1
    return data

a=data_Noise()
#noise = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.25]
# noise=[0]
#noiseEnd = [0, 1, 3, 5, 8, 10]
#n=[1,3,5,8,10]
n=[0, 1, 3, 5, 8, 10]
x=np.array([j for j in range(1,21)])

plt.figure(dpi=200)
ax = plt.gca()
for index,i in enumerate(a):
    if index==0:
        plt.plot(x, i, label="without noise" )
    else:
        plt.plot(x, i, label="Translation Noise Level %s mm, Orientation Noise (1 degree) " % (n[(index % 6)]))

plt.ylim(0, 20)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(loc='upper right',fontsize=9)
# Add title and x, y labels
#plt.title("Orientation Error H1 Hypothesis", fontsize=10)
#plt.suptitle("Random Walk Suptitle", fontsize=10)
plt.xlabel("Epoch",fontsize=20)
plt.ylabel("RMSE(degree)",fontsize=20)

plt.savefig("H1OrientationPPPPPPPPPPPPPP.svg")
plt.show()
