import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)
def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)


def read_data_box():
    data = list()
    myp = "/media/ozanba/Seagate Portable Drive/RWCL/CalibrationwithImages_RWE_Blue_robot_roomCL/camerapose.txt"
    with open(myp,'r') as f:
        for lindex,line in enumerate(f):
            linedata = line.split()

            temp=list()
            if lindex%100==0:
                for index,j in enumerate(linedata):
                    temp.append(float(j))
                data.append(temp)
    return data

tt=read_data_box()
dt=np.asarray(tt)
asassasasas=3
setattr(Axes3D, 'annotate3D', _annotate3D)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.axes.set_zlim3d(bottom=0.2, top=0.8)
lsttest = [5, 3, 8, 15, 21, 19]
colors = ['b', 'c', 'y', 'm', 'r']
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for index,i in enumerate(dt):

    idf=index+1
    if idf in lsttest:
        ax.scatter(i[0], i[1], i[2], marker='o', color='b')
    else:
        ax.scatter(i[0], i[1], i[2], marker='o', color='r')
    ax.annotate3D(str(idf), (i[0], i[1], i[2]), xytext=(3, 3), textcoords='offset points')
    #plt.legend(index)

ax.set_title("Camera Configuration in Real-world")
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
plt.show()


'''
data10Dm=dt[:,3]
data10Ds=dt[:,4]
dataPm=dt[:,5]
dataPs=dt[:,6]
# Creating dataset
np.random.seed(10)

data_1 = np.random.normal(100, 10, 200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)
data = [data10Dm, data10Ds, dataPm, dataPs]

fig = plt.figure(figsize=(10, 7))

# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])

# Creating plot
bp = ax.boxplot(data)

# show plot
plt.show()
'''