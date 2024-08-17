<p align="center">
  <h1 align="center">Gradient-NBV: Gradient-based Local Next-best-view Planning for Improved Perception of Targeted Plant Nodes</h1>
  <p align="center">
    <strong>Akshay K. Burusa</strong>
    ·
    <strong>Eldert J. van Henten</strong>
    ·
    <strong>Gert Kootstra</strong>
  </p>
</p>

https://github.com/akshaykburusa/gradientnbv/assets/127020264/dfa1f2a9-f07c-4af0-84ef-7ea20a7cb61b

## Installation

### Prerequisites

[Ubuntu 20.04](https://releases.ubuntu.com/20.04/)  
[ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)  
At least 8GB of GPU VRAM

### Clone the repository

```
git clone https://github.com/akshaykburusa/gradientnbv.git
```

### ROS dependencies

Quick install:
```
cd gradient_nbv
sudo ./install_ros_pkgs.sh
```
Manual install:
```
sudo apt install ros-noetic-moveit
sudo apt install ros-noetic-ros-controllers
sudo apt install ros-noetic-trac-ik
```

### Python packages

Quick install:
```
cd gradient_nbv
conda env create -f install_conda_pkgs.yaml
```
Manual install:
```
conda create -n grad_nbv python==3.8.10
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pyyaml==6.0.1
pip install rospkg==1.5.1
pip install opencv-python==4.9.0.80
pip install scipy==1.10.1
pip install pytransform3d==3.5.0
pip install open3d==0.18.0
```

### Compile
```
cd gradient_nbv
catkin_make -DCMAKE_BUILD_TYPE=Release
```

## Execute

Bring up the simulation environment with the robot (ABB robotic arm + Realsense L515 camera):
```
roslaunch abb_l515_bringup abb_l515_bringup.launch
```

Start the gradient-based local next-best-view planner in a new terminal:
```
conda activate grad_nbv
roslaunch viewpoint_planning viewpoint_planning.launch
```

## Citation
```bibtex
@inproceedings{burusa2024gradient,
  title={Gradient-based local next-best-view planning for improved perception of targeted plant nodes},
  author={Burusa, Akshay K and van Henten, Eldert J and Kootstra, Gert},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={15854--15860},
  year={2024},
  organization={IEEE}
}
```
