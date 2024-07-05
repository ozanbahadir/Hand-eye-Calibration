from itertools import chain
import random
from torch.utils.data import Subset
import numpy as np
import torch
import dataset as dt
from torch.utils.data import TensorDataset, DataLoader
import itertools
def print_text(Resultt,loss_stats):
    
    Resultt.write(str(loss_stats[0][-1]) + "\t")
    Resultt.write(str(loss_stats[1][-1]) + "\t")
    Resultt.write(str(loss_stats[2][-1]) + "\t")
    Resultt.write(str(loss_stats[3][-1]) + "\t")
    Resultt.write(str(loss_stats[4][-1]) + "\t")
    Resultt.write(str(loss_stats[5][-1]) + "\t")
    Resultt.write(str(loss_stats[6][-1]) + "\n")
   
def train_val_datasetCN(dataset, robotname,h,buf,idxlst):
    torch.manual_seed(1)



    lstTest, lstTrain = CL_split6()
    test_idx1 = list(chain.from_iterable(lstTest[0]))
    test_idx2 = list(chain.from_iterable(lstTest[1]))
    test_idx3 = list(chain.from_iterable(lstTest[2]))
    test_idx4 = list(chain.from_iterable(lstTest[3]))
    test_idx5 = list(chain.from_iterable(lstTest[4]))
    test_idx6 = list(chain.from_iterable(lstTest[5]))
    train_idx = list(chain.from_iterable(lstTrain[h]))

    datasets = {}

    
    datasets['test1'] = Subset(dataset, test_idx1)
    datasets['test2'] = Subset(dataset, test_idx2)
    datasets['test3'] = Subset(dataset, test_idx3)
    datasets['test4'] = Subset(dataset, test_idx4)
    datasets['test5'] = Subset(dataset, test_idx5)
    datasets['test6'] = Subset(dataset, test_idx6)
    if h != 0 and buf:
        temp = list()

        for kk in range(h):
            # if h != 5:
            if idxlst[kk]==False:
                rndCL = [i for i in range(len(lstTrain[kk]))]
             

                bf = lstTrain[kk]

                for indexi, i in enumerate(bf):
                 

                    rnd1 = random.sample(i, k=4)
                   
                    temp.append(rnd1)
       

      
        buffer_idx = list(chain.from_iterable(temp))
        datasets['buffer'] = Subset(dataset, buffer_idx)
        
        datasets['train'] = Subset(dataset, train_idx)
     
    else:
        datasets['train'] = Subset(dataset, train_idx)
        aaaa = 5
    return datasets

def train_val_datasetN(dataset, robotname,h,buf,idxlst):
    torch.manual_seed(1)



    lstTest, lstTrain = CL_split5()
    #test_idx1 = list(chain.from_iterable(lstTest[0]))
    #test_idx2 = list(chain.from_iterable(lstTest[1]))
    #test_idx3 = list(chain.from_iterable(lstTest[2]))
   
    train_idx = list(chain.from_iterable(lstTrain[h]))

    datasets = {}


    #datasets['test1'] = Subset(dataset, test_idx1)
    #datasets['test2'] = Subset(dataset, test_idx2)
    #datasets['test3'] = Subset(dataset, test_idx3)
 
    if h != 0 and buf:
        temp = list()

        for kk in range(h):
          
            if idxlst[kk]==False:
                rndCL = [i for i in range(len(lstTrain[kk]))]
          
                bf = lstTrain[kk]

                for indexi, i in enumerate(bf):
                 

                    rnd1 = random.sample(i, k=4)
                 
                    temp.append(rnd1)
       

     
        buffer_idx = list(chain.from_iterable(temp))
        datasets['buffer'] = Subset(dataset, buffer_idx)
     
        datasets['train'] = Subset(dataset, train_idx)
     
    else:
        datasets['train'] = Subset(dataset, train_idx)
    return datasets



def train_val_dataset(dataset, robotname,h,buf):
  
    torch.manual_seed(1)
    if robotname=='CL' or robotname=='ur3qrCL':
        if robotname=='CL':
            lstTest,lstTrain=CL_split3()



            test_idx1 = list(chain.from_iterable(lstTest[0]))
            test_idx2 =  list(chain.from_iterable(lstTest[1]))
            test_idx3 =  list(chain.from_iterable(lstTest[2]))
            test_idx4 =  list(chain.from_iterable(lstTest[3]))
            test_idx5 =  list(chain.from_iterable(lstTest[4]))
            test_idx6 =  list(chain.from_iterable(lstTest[5]))
            train_idx = list(chain.from_iterable(lstTrain[h]))


            datasets = {}

            datasets['train'] = Subset(dataset, train_idx)
            datasets['test1'] = Subset(dataset, test_idx1)
            datasets['test2'] = Subset(dataset, test_idx2)
            datasets['test3'] = Subset(dataset, test_idx3)
            datasets['test4'] = Subset(dataset, test_idx4)
            datasets['test5'] = Subset(dataset, test_idx5)
            datasets['test6'] = Subset(dataset, test_idx6)

            if h !=0 and buf:
                temp = list()

                for kk in range(h):
              
                    rndCL = [i for i in range(len(lstTrain[kk]))]
                

                

                    bf= lstTrain[kk]

                    for indexi,i in enumerate(bf):
                     
                        rnd1 = random.sample(i, k=4)
                      
                        temp.append(rnd1)
              

           
                buffer_idx = list(chain.from_iterable(temp))
                datasets['buffer'] = Subset(dataset, buffer_idx)
                
                datasets['train'] = Subset(dataset, train_idx)
              
            else:
                datasets['train'] = Subset(dataset, train_idx)
            aaaa=5
        else:
            lstTest, lstTrain = CL_split4()
            test_idx1 = list(chain.from_iterable(lstTest[0]))
            test_idx2 = list(chain.from_iterable(lstTest[1]))
            test_idx3 = list(chain.from_iterable(lstTest[2]))
       
            train_idx = list(chain.from_iterable(lstTrain[h]))

            datasets = {}

         
            datasets['test1'] = Subset(dataset, test_idx1)
            datasets['test2'] = Subset(dataset, test_idx2)
            datasets['test3'] = Subset(dataset, test_idx3)

        if h != 0 and buf:
            temp = list()

            for kk in range(h):
           
                rndCL = [i for i in range(len(lstTrain[kk]))]
            
                bf = lstTrain[kk]

                for indexi, i in enumerate(bf):
                 
                    rnd1 = random.sample(i, k=4)
                
                    temp.append(rnd1)
         )
            buffer_idx = list(chain.from_iterable(temp))
            datasets['buffer'] = Subset(dataset, buffer_idx)
           
            datasets['train'] = Subset(dataset, train_idx)
          
        else:
            datasets['train'] = Subset(dataset, train_idx)
        aaaa = 5


    else:

        if robotname=='ur3':
            lst = [i for i in range(1, 25)]
            rnd = [17, 19, 9, 8, 18]
            nimage=100
            nimage = [100 for i in range(24)]
        elif robotname=='baxter':
            lst = [i for i in range(1, 20)]
            rnd = [16, 17, 18, 19]
            nimage = 85
            nimage = [85 for i in range(19)]
        elif robotname=='ur3qr':
            lst = [i for i in range(1, 25)]
           
            rnd = random.sample(lst, k=5)
            rnd = [2,8,10,18,22]
            print(rnd)

            nimage = 100
            nimage = [i*100 for i in range(25)]

        else:
            lst = [i for i in range(1, 109)]
            rnd = random.sample(lst, k=30)
           
            nimage = [0, 45, 93, 142, 190, 237, 285, 334, 384, 428, 475, 522, 571, 620, 670, 720, 767, 816, 866, 916, 966, 1016,
         1066, 1115, 1163, 1212, 1261, 1310, 1359, 1406, 1454, 1502, 1545, 1593, 1640, 1687, 1735, 1783, 1830, 1877,
         1924, 1972, 2021, 2065, 2112, 2162, 2207, 2257, 2306, 2355, 2404, 2453, 2502, 2546, 2595, 2644, 2693, 2742,
         2790, 2839, 2888, 2937, 2985, 3033, 3082, 3126, 3172, 3220, 3268, 3311, 3358, 3402, 3450, 3497, 3546, 3587,
         3634, 3681, 3728, 3777, 3821, 3868, 3917, 3960, 4009, 4056, 4104, 4151, 4198, 4245, 4294, 4341, 4388, 4437,
         4480, 4526, 4575, 4622, 4669, 4716, 4763, 4812, 4858, 4905, 4953, 5000, 5048, 5096, 5137]
    
        test_idx1 = list()
        train_idx1 = list()
      
        print(rnd)
       
        for i in lst:
            if i in rnd:
               
                test_idx1.append([j for j in range(nimage[i - 1], nimage[i])])
            else:
                
                train_idx1.append([j for j in range(nimage[i - 1], nimage[i])])

        test_idx = list(chain.from_iterable(test_idx1))
        train_idx = list(chain.from_iterable(train_idx1))
        
        
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)
        datasets['test'] = Subset(dataset, test_idx)

    return datasets



def read_data(robotname,cr):

    knownTrans=np.asarray(dt.read_input(robotname)[0])#*1000

    if (cr):
        dataset_im = dt.read_image(robotname)
    else:
        dataset_im=False
    cam=np.asarray(dt.camera_pose(robotname))
    cam2ref=np.asarray(dt.cam2ref(robotname)[0])

    return dataset_im,knownTrans,cam,cam2ref

def merge_data(im,knw,cam,hypoth,num_classes,cam2ref):
  

    dataset_im = im
    if hypoth == 3:
        knownTrans = knw
        if num_classes == 3:
        
            label = cam[:, :3] 
            ggg = 5
        elif num_classes == 4:
         
            label = cam[:, 3:]
        elif num_classes == 10:
         
            label = cam[:, 3:]
        else:
            label = cam

    elif hypoth == 1:
        knownTrans = np.concatenate((knw, cam2ref), axis=1)
      
        if num_classes == 3:
       
            label = cam[:, :3]
            ggg = 5
        elif num_classes == 4:
         
            label = cam[:, 3:]
        else:
            label = cam
    else:
        knownTrans = knw
        if num_classes == 3:
            label = cam2ref[:, :3]
          
        else:
            label = cam2ref[:, 3:]
           


    knownTrans = torch.from_numpy(knownTrans)
    label = torch.from_numpy(label)
    if dataset_im==False:
        Data_set = TensorDataset(knownTrans, label)

    else:
        data1 = torch.from_numpy(np.asarray(dataset_im[0]))
        data2 = torch.from_numpy(np.asarray(dataset_im[1]))
        Data_set = TensorDataset(data1, data2, knownTrans, label)

    return Data_set


def CL_split():
    lst = [i for i in range(1, 109)]
    all_list=[[],[],[],[],[],[]]

    for index,i in enumerate(lst):
        if index%36<6:
            all_list[0].append(i)
        elif index%36>=6 and index%36<12:
            all_list[1].append(i)
        elif index%36>=12 and index%36<18:
            all_list[2].append(i)
        elif index%36>=18 and index%36<24:
            all_list[3].append(i)
        elif index%36>=24 and index%36<30:
            all_list[4].append(i)
        else:
            all_list[5].append(i)

    return all_list

def CL_split2():
    nimage=50

    nimageL = [0, 45, 93, 142, 190, 237, 285, 334, 384, 428, 475, 522, 571, 620, 670, 720, 767, 816, 866, 916, 966, 1016,
         1066, 1115, 1163, 1212, 1261, 1310, 1359, 1406, 1454, 1502, 1545, 1593, 1640, 1687, 1735, 1783, 1830, 1877,
         1924, 1972, 2021, 2065, 2112, 2162, 2207, 2257, 2306, 2355, 2404, 2453, 2502, 2546, 2595, 2644, 2693, 2742,
         2790, 2839, 2888, 2937, 2985, 3033, 3082, 3126, 3172, 3220, 3268, 3311, 3358, 3402, 3450, 3497, 3546, 3587,
         3634, 3681, 3728, 3777, 3821, 3868, 3917, 3960, 4009, 4056, 4104, 4151, 4198, 4245, 4294, 4341, 4388, 4437,
         4480, 4526, 4575, 4622, 4669, 4716, 4763, 4812, 4858, 4905, 4953, 5000, 5048, 5096, 5137]

   
    lsttest =[[3, 41, 74, 4], [8, 47, 80, 84], [14, 50, 89, 16], [20, 57, 93, 22], [28, 63, 101, 29], [32, 68, 106, 31]]
    lstall=[[1, 2, 3, 4, 5, 6, 37, 38, 39, 40, 41, 42, 73, 74, 75, 76, 77, 78], [7, 8, 9, 10, 11, 12, 43, 44, 45, 46, 47, 48, 79, 80, 81, 82, 83, 84], [13, 14, 15, 16, 17, 18, 49, 50, 51, 52, 53, 54, 85, 86, 87, 88, 89, 90], [19, 20, 21, 22, 23, 24, 55, 56, 57, 58, 59, 60, 91, 92, 93, 94, 95, 96], [25, 26, 27, 28, 29, 30, 61, 62, 63, 64, 65, 66, 97, 98, 99, 100, 101, 102], [31, 32, 33, 34, 35, 36, 67, 68, 69, 70, 71, 72,103,104,105,106,107,108]]
  
    lstTestA=list()
    lstTrainA = list()
    lstTrain=[[1, 2, 5, 6, 37, 38, 39, 40, 42, 73, 75, 76, 77, 78], [7, 9, 10, 11, 12, 43, 44, 45, 46, 48, 79, 81, 82, 83], [13, 15, 17, 18, 49, 51, 52, 53, 54, 85, 86, 87, 88, 90], [19, 21, 23, 24, 55, 56, 58, 59, 60, 91, 92, 94, 95, 96], [25, 26, 27, 30, 61, 62, 64, 65, 66, 97, 98, 99, 100, 102], [33, 34, 35, 36, 67, 69, 70, 71, 72, 103, 104, 105, 107, 108]]

    for i in lsttest:
        temp=list()
        for j in i:
            temp.append([k for k in range(nimageL[j - 1], nimageL[j])])
        
        lstTestA.append(temp)
  

    for i in lstTrain:
        temp=list()
        for j in i:
           )
            temp.append([k for k in range(nimageL[j - 1], nimageL[j])])
         
        lstTrainA.append(temp)
 


    return lstTestA, lstTrainA

def CL_split3():
    nimage=50
    
    nimageL=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31, 32, 33, 34, 35, 36]
   
    lsttest =[[4], [8], [14], [18], [27], [34]]
    
    lstall=[[1, 2, 3, 4, 5, 6, 37, 38, 39, 40, 41, 42, 73, 74, 75, 76, 77, 78], [7, 8, 9, 10, 11, 12, 43, 44, 45, 46, 47, 48, 79, 80, 81, 82, 83, 84], [13, 14, 15, 16, 17, 18, 49, 50, 51, 52, 53, 54, 85, 86, 87, 88, 89, 90], [19, 20, 21, 22, 23, 24, 55, 56, 57, 58, 59, 60, 91, 92, 93, 94, 95, 96], [25, 26, 27, 28, 29, 30, 61, 62, 63, 64, 65, 66, 97, 98, 99, 100, 101, 102], [31, 32, 33, 34, 35, 36, 67, 68, 69, 70, 71, 72,103,104,105,106,107,108]]
    
    lstTestA=list()
    lstTrainA = list()
    
    lstTrain = [[0, 1, 2, 3, 5], [6, 7, 9, 10, 11], [12, 13, 15, 16, 17], [19, 20, 21, 22, 23], [24, 25, 26, 28, 29],
                [30, 31, 32, 33, 35]]
    for i in lsttest:
        temp=list()
        for j in i:
            temp.append([k for k in range(nimageL[j - 1], nimageL[j])])
           
        lstTestA.append(temp)
 

    for i in lstTrain:
        temp=list()
        for j in i:
           
            temp.append([k for k in range(nimageL[j - 1], nimageL[j])])
          
        lstTrainA.append(temp)
   


    return lstTestA, lstTrainA
def CL_split4():
    nimage=100
   

    nimageL = [i * 100 for i in range(25)]
   
    lsttest =[[4,2], [7,14], [20,18]]
    lsttest = [[5, 3], [8, 15], [21, 19]]
    #l
    lstall=[[1, 2, 3, 4, 5, 6, 37, 38, 39, 40, 41, 42, 73, 74, 75, 76, 77, 78], [7, 8, 9, 10, 11, 12, 43, 44, 45, 46, 47, 48, 79, 80, 81, 82, 83, 84], [13, 14, 15, 16, 17, 18, 49, 50, 51, 52, 53, 54, 85, 86, 87, 88, 89, 90], [19, 20, 21, 22, 23, 24, 55, 56, 57, 58, 59, 60, 91, 92, 93, 94, 95, 96], [25, 26, 27, 28, 29, 30, 61, 62, 63, 64, 65, 66, 97, 98, 99, 100, 101, 102], [31, 32, 33, 34, 35, 36, 67, 68, 69, 70, 71, 72,103,104,105,106,107,108]]
    
    lstTestA=list()
    lstTrainA = list()
    
    lstTrain = [[0, 1, 3, 5,6], [8,9,10,13,15,16], [11,12,17,19,21,22,23]]
    lstTrain = [[1, 2, 4, 6, 7], [9, 10, 11, 14, 16, 17], [12, 13, 18, 20, 22, 23, 24]]
    for i in lsttest:
        temp=list()
        for j in i:
            temp.append([k for k in range(nimageL[j - 1], nimageL[j])])
            
        lstTestA.append(temp)
    

    for i in lstTrain:
        temp=list()
        for j in i:
           
            temp.append([k for k in range(nimageL[j - 1], nimageL[j])])
         
        lstTrainA.append(temp)
   


    return lstTestA, lstTrainA


def CL_split5():



    nimageL = [i * 100 for i in range(25)]
    lsttest = [[5, 3], [8, 15], [21, 19]]


    lstTestA=list()
    lstTrainA = list()

    lstTrain = [[1, 2, 4, 6, 7], [9], [10], [11], [14], [16], [17], [12], [13], [18], [20], [22], [23], [24]]
    for i in lsttest:
        temp=list()
        for j in i:
            temp.append([k for k in range(nimageL[j - 1], nimageL[j])])

        lstTestA.append(temp)


    for i in lstTrain:
        temp=list()
        for j in i:

            temp.append([k for k in range(nimageL[j - 1], nimageL[j])])

        lstTrainA.append(temp)




    return lstTestA, lstTrainA



def CL_split6():
    nimage = 50
   
    nimageL = [0, 45, 93, 142, 190, 237, 285, 334, 384, 428, 475, 522, 571, 620, 670, 720, 767, 816, 866, 916, 966,
               1016,1066, 1115, 1163, 1212, 1261, 1310, 1359, 1406, 1454, 1502, 1545, 1593, 1640, 1687, 1735, 1783, 1830,
               1877,1924, 1972, 2021, 2065, 2112, 2162, 2207, 2257, 2306, 2355, 2404, 2453, 2502, 2546, 2595, 2644, 2693,
               2742,2790, 2839, 2888, 2937, 2985, 3033, 3082, 3126, 3172, 3220, 3268, 3311, 3358, 3402, 3450, 3497, 3546,
               3587,3634, 3681, 3728, 3777, 3821, 3868, 3917, 3960, 4009, 4056, 4104, 4151, 4198, 4245, 4294, 4341, 4388,
               4437,4480, 4526, 4575, 4622, 4669, 4716, 4763, 4812, 4858, 4905, 4953, 5000, 5048, 5096, 5137]

    
    lsttest = [[3, 41, 74, 4], [8, 47, 80, 84], [14, 50, 89, 16], [20, 57, 93, 22], [28, 63, 101, 29],[32, 68, 106, 31]]
    lstall = [[1, 2, 3, 4, 5, 6, 37, 38, 39, 40, 41, 42, 73, 74, 75, 76, 77, 78],
              [7, 8, 9, 10, 11, 12, 43, 44, 45, 46, 47, 48, 79, 80, 81, 82, 83, 84],
              [13, 14, 15, 16, 17, 18, 49, 50, 51, 52, 53, 54, 85, 86, 87, 88, 89, 90],
              [19, 20, 21, 22, 23, 24, 55, 56, 57, 58, 59, 60, 91, 92, 93, 94, 95, 96],
              [25, 26, 27, 28, 29, 30, 61, 62, 63, 64, 65, 66, 97, 98, 99, 100, 101, 102],
              [31, 32, 33, 34, 35, 36, 67, 68, 69, 70, 71, 72, 103, 104, 105, 106, 107, 108]]
    
    lstTestA = list()
    lstTrainA = list()
    lstTrain = [[1, 2, 5, 6, 37, 38, 39, 40, 42, 73, 75, 76, 77, 78],[7], [9], [10], [11], [12], [43], [44], [45], [46], [48], [79], [81], [82], [83], [13], [15], [17],
                [18], [49],[51], [52], [53], [54], [85], [86], [87], [88], [90], [19], [21], [23], [24], [55], [56], [58], [59],[60], [91], [92], [94], [95], [96], [25], [26], [27], [30], [61], [62], [64], [65], [66], [97], [98], [99], [100],
                [102], [33],[34], [35], [36], [67], [69], [70], [71], [72], [103], [104], [105], [107], [108]]


    for i in lsttest:
        temp = list()
        for j in i:
            temp.append([k for k in range(nimageL[j - 1], nimageL[j])])
            
        lstTestA.append(temp)
   

    for i in lstTrain:
        temp = list()
        for j in i:
            
            temp.append([k for k in range(nimageL[j - 1], nimageL[j])])
           
        lstTrainA.append(temp)
    

    return lstTestA, lstTrainA
