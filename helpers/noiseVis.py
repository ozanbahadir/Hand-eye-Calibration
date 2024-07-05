import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

mypath = Path().absolute().parent
print(mypath)

def data_Noise():
    data = list()
    #with open(str(mypath) + "/Result/TransNoiseWithEndEffandImage.txt",'r') as f:
    #with open(str(mypath) + "/Result/H2TranslationWithNoise.txt", 'r') as f:
    #with open(str(mypath) + "/Result/H2OrientationWithNoise.txt", 'r') as f:
    with open(str(mypath) + "/Result/OrientationResult030421.txt", 'r') as f:
        c=0
        for line in f:
            linedata = line.split()

            temp=list()
            if c %2 ==1:
                for index,j in enumerate(linedata):
                    temp.append(math.degrees(math.sqrt(float(j))))
                    #temp.append(math.degrees(float(j)))
                    #temp.append(math.sqrt(float(j)*1000))
                data.append(temp)
            c+=1
    return data

a=data_Noise()
#noise = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.25]
# noise=[0]
#noiseEnd = [0, 1, 3, 5, 8, 10]
#n=[1,3,5,8,10]
'''
minV=list()
for i in a:
    minV.append(min(i))

iii=32
Resultt = open(str(mypath) + "/Result/" + "H2andH3ForTable.txt", "a")
print_text(Resultt,minV)
'''
n=[0, 1, 3, 5, 8, 10]
x=np.array([j for j in range(1,16)])

plt.figure(dpi=300)
#plt.figure(figsize=(20,20))
fig, ax = plt.subplots(nrows=3, ncols=3,  figsize=(20,20))

fig.text(0.5, 0.04, 'Epoch', ha='center',fontsize=38)
fig.text(0.04, 0.5, 'RMSE(mm)', va='center', rotation='vertical',fontsize=38)

for index,i in enumerate(a):
    if index < 6:
        plt.subplot(3, 3, 1)
        plt.plot(x, i,label="Without Noise " )#, label="Translation Noise Level %s mm, Orientation Noise (1 degree)" % (n[index%5]))
        plt.title("ImageNoise(0%)",fontsize=24)
    elif index >=6 and index<12:
        plt.subplot(3, 3, 2)
        plt.plot(x, i ,label="Translation Noise Level %s mm, Orientation Noise (1 degree)" % (n[index%6]))#label="Translation Noise Level %s mm, Orientation Noise (1 degree)" % (n[index % 5]))
        plt.title("ImageNoise(1%)",fontsize=24)
    elif index >=12 and index<18:
        plt.subplot(3, 3, 3)
        plt.plot(x, i,label="Translation Noise Level %s mm, Orientation Noise (1 degree)" % (n[index%6]))
        plt.title("ImageNoise(2%)",fontsize=24)
    elif index >=18 and index<24:
        plt.subplot(3, 3, 4)
        plt.plot(x, i ,label="Translation Noise Level %s mm, Orientation Noise (1 degree)" % (n[index%6]))
        plt.title("ImageNoise(5%)",fontsize=24)
    elif index >=24 and index<30:
        plt.subplot(3, 3, 5)
        plt.plot(x, i ,label="Translation Noise Level %s mm, Orientation Noise (1 degree)" % (n[index%6]))
        plt.title("ImageNoise(8%)",fontsize=24)
    elif index >=30 and index<36:
        plt.subplot(3, 3, 6)
        plt.plot(x, i ,label="Translation Noise Level %s mm, Orientation Noise (1 degree)" % (n[index%6]))
        plt.title("ImageNoise(10%)",fontsize=24)
    elif index >=36 and index<42:
        plt.subplot(3, 3, 7)
        plt.plot(x, i ,label="Translation Noise Level %s mm, Orientation Noise (1 degree)" % (n[index%6]))
        plt.title("ImageNoise(25%)",fontsize=24)
'''    
    
    elif index >=35 and index<40:
        plt.subplot(5, 2, 8)
        plt.plot(x, i ,label="Translation Noise Level %s mm, Orientation Noise (1 degree)" % (n[index%5]))
        plt.title("ImageNoise(50%)")
    elif index >=40 and index<45:
        plt.subplot(5, 2, 9)
        plt.plot(x, i,label="Translation Noise Level %s mm, Orientation Noise (1 degree)" % (n[index%5]) )
        plt.title("ImageNoise(70%)")
    else:
        plt.subplot(5, 2, 10)
        plt.plot(x, i ,label="Translation Noise Level %s mm, Orientation Noise (1 degree)" % (n[index%5]))
        plt.title("ImageNoise(100%)")
'''
plt.ylim(0, 30)

ax[2,2].axis('off')
ax[2,1].axis('off')
#plt.xlabel('Epoch')
# Set the y axis label of the current axis.
#plt.ylabel('MSE(mm)')
#plt.ylim(0, 30)
#plt.legend(bbox_to_anchor=(1.0,1.0),\
  #  bbox_transform=plt.gcf().transFigure)

#plt.set_xlabel('Epoch')
# Set the y axis label of the current axis.
#plt.set_ylabel('MSE(mm)')

lines, labels = fig.axes[-1].get_legend_handles_labels()
#plt.tick_params(bottom="off", left="off")
fig.legend(lines, labels, loc = (0.40, 0.15),prop={'size': 24},frameon=False)
plt.savefig("H1Translationpppppppppppp.svg")
plt.show()
#print(math.degrees(0.05))
ll=5

