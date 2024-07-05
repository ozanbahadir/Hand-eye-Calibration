# Hand-Eye Calibration Using Deep Learning
This repository provides a deep learning approach for hand-eye calibration using synthetic and real-world datasets. The project includes models for training and testing, with a focus on continuous learning and evaluation.

## Features
  - Synthetic Dataset Generation: Synthetic data can be generated using the scripts in the Simulation folder.
  - Real-World Data Integration: Follow the data preparation guidelines in the documentation to incorporate real-world datasets.
  - Training and Evaluation: Models can be trained and evaluated using different hyperparameters and noise levels.

## Models
  - **RotationNet:** For orientation-related tasks.
  - **UnetR:** For translation-related tasks.
 
## Requirements
  - Python
  - PyTorch
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-image
  - Seaborn
## Usage
  - **Set Up Environment:** Ensure all dependencies are installed.
  - **Generate Synthetic Data:** Use the scripts in the Simulation folder.
  - **Prepare Real-World Data:** Follow the guidelines in the documentation.
  - **Train Models:** Adjust hyperparameters as needed and run training scripts.
  - **Evaluate Models:** Use the provided evaluation functions.
## Run
'sh python RegressionWithImage.py' for Hypotheses 2 and 3
'sh python Ref_Net_FNNS.py' for Hypotheses 1
