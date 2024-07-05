# Hand-eye-Calibration for Robots

This repository contains implementations and experiments for hand-eye calibration in robotic systems, leveraging continual learning and deep learning approaches.

## Overview
Hand-eye calibration is a critical task in robotics that ensures accurate coordination between a robot's visual sensors and its manipulators. This project explores two advanced methodologies:

### Continual Learning Approaches to Hand-Eye Calibration

This method focuses on enabling robots to adapt their calibration over time without requiring complete retraining from scratch, thus maintaining performance and adapting to changes dynamically.

### Deep Learning-Based Hand-Eye Calibration Using a Single Reference Point

This approach utilizes deep neural networks to perform hand-eye calibration, improving efficiency and reducing setup time.

#### Features
Continual Learning: Implementations of various continual learning algorithms that allow the robot to update its calibration as it encounters new data.
Deep Learning Models: Pre-trained models and training scripts for hand-eye calibration using a single reference point.
Evaluation Metrics: Tools to evaluate the performance of calibration methods, including accuracy and robustness metrics.
Installation
Clone the repository and install the required dependencies:



## Description
This repository enables you to collect virtual data in the pybullet environment by using the UR5 robot equipped with a parallel gripper.
## Getting Started
clone the following repository into your workspace:
```
cd /simulation
git clone https://github.com/Alchemist77/pybullet-ur5-equipped-with-robotiq-140.git
```
```
pip3 install pybullet
```


### Dependencies
* Ubuntu 16.04
* python3
### Installing

* Replace ur5_robotiq140.py with run_ur5_robotiq140 .py 

### Executing program

* open a terminal
```
python3 run_ur5_robotiq140.py
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

Ozan Bahadir  
contact info: ozanbahadir61@gmail.com

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Results




## Acknowledgments

Inspiration, code snippets, etc.
* [Alchemist77](https://github.com/Alchemist77/pybullet-ur5-equipped-with-robotiq-140/)






### Executing program

* You can use ethier simulation or real-world data collection 
* Then you can train your model by following step
* open a terminal 
```
python3 .py 
```
