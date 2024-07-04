## Description
This repository contains a Python script for simulating a UR5 robot with a Robotiq 140 gripper in a PyBullet environment. The script randomly generates positions and orientations for the robot's end-effector, captures RGB and RGBD images, and saves these images along with relevant data to files.
## Requirements

- Python 3.7+
- PyBullet
- NumPy
- Matplotlib
- Pickle
- Pillow


## Installation

1. Clone the repository:

```sh
git clone https://github.com/ozanbahadir/Hand-eye-Calibration.git
```
2. Install the required Python packages:
```
pip install pybullet numpy matplotlib pillow

```

## Usage
1. Ensure you have the required URDF files for the UR5 robot and Robotiq 140 gripper. Place these files in the appropriate directories as specified in the script.

2. Run the script:
```
python data_generation_CL.py
```
The script will:

1. Initialize the PyBullet environment.
2. Load the UR5 robot with the Robotiq 140 gripper.
3. Position a stereo pair of camera sin 108 different configurations using sine and cosine functions with a distance relative to the robot base
4. Randomly generate positions and orientations for the robot's end-effector.
5. Move the robot's end-effector to the generated positions.
6. Capture RGB and RGBD images at each position.
7. Save the images and relevant data to files.

## Output
The script generates the following output files:

- Images/rgb_image/rgb_imageX.pkl: RGB images captured at each position.
- Images/rgb_image/rgbd_imageX.pkl: RGBD images captured at each position.
- Images/input.txt1: End-effector positions and orientations.
- Images/target_pixel1.txt: Screen coordinates of the end-effector.
- Images/camera_pose1.txt: Camera positions and orientations.

## Customization
You can customize the script by adjusting the following parameters:

- sample_number: Number of random samples (positions and orientations) to generate.
- ranges: Range of values for the end-effector positions.
- ranges1: Range of values for the end-effector orientations.
-view_matrix and projection_matrix: Parameters for the camera view and projection.

## Acknowledgements
This project uses the following libraries:

PyBullet
NumPy
Matplotlib
Pillow

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

## Acknowledgments

Inspiration, code snippets, etc.
* [Alchemist77](https://github.com/Alchemist77/pybullet-ur5-equipped-with-robotiq-140/)


