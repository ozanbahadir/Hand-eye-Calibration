#Description
The **CL_HEC1.py** script implements the Continual Learning-based HEC approach using the Naive method and the reservoir buffer with class balance methods. The **CL_HEC2.py** script implements the Continual Learning-based HEC approach using The reservoir buffer with class balance and camera pose selection.

Setting **buf=True** activates the reservoir buffer with class balance approach; otherwise, it defaults to the Naive CL method. The **robotname='CL'** parameter is pivotal as it partitions the dataset into subsets following the principles of CL methods.
