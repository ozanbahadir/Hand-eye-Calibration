# Hand-eye-Calibration for Robots

This repository contains implementations and experiments for hand-eye calibration in robotic systems, leveraging continual learning and deep learning approaches.

## Overview
Hand-eye calibration is a critical task in robotics that ensures accurate coordination between a robot's visual sensors and its manipulators. This project explores two advanced methodologies:

### 1. Continual Learning Approaches to Hand-Eye Calibration

This method focuses on enabling robots to adapt their calibration over time without requiring complete retraining from scratch, thus maintaining performance and adapting to changes dynamically.

### 2. Deep Learning-Based Hand-Eye Calibration Using a Single Reference Point

This approach utilizes deep neural networks to perform hand-eye calibration, improving efficiency and reducing setup time.

## Features
Continual Learning: Implementations of various continual learning algorithms that allow the robot to update its calibration as it encounters new data.
Deep Learning Models: Pre-trained models and training scripts for hand-eye calibration using a single reference point.

Evaluation Metrics: Tools to evaluate the performance of calibration methods, including accuracy and robustness metrics.

## Installation
Clone the repository and install the required dependencies:


## Dataset
The project employs a synthetic dataset for training and testing purposes. To generate synthetic data, please refer to the **Simulation** folder within the repository.

Additionally, real-world datasets can be incorporated by adhering to the data preparation guidelines outlined in the documentation.


## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Authors

Contributors names and contact info

Dr Ozan Bahadir  
contact info: ozanbahadir61@gmail.com, obahadir@kho.msu.edu.tr

Dr Gerardo Aragon Camarasa
contact info:Gerardo.AragonCamarasa@glasgow.ac.uk

Dr Jan Paul Siebert 
contact info:Paul.Siebert@glasgow.ac.uk


## Acknowledgements
This work is based on the research presented in the following papers:

-Continual Learning Approaches to Hand-Eye Calibration in Robots"
-A Deep Learning-Based Hand-Eye Calibration Approach Using a Single Reference Point on a Robot Manipulator" [paper link] (https://ieeexplore.ieee.org/abstract/document/10011774)
