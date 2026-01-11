# Planning and Decision-Making project for Group 27
This repository contains a navigation system for a four-wheeled mobile robot using a kinematic bicycle model. The project is split into two main components:
- RRT*
- MPC

## Repository Structure
```text
Project
├── MPC/
│   ├── Config.py
│   ├── Env_test.py
│   ├── Environment.py
│   ├── MPC_core.py
│   ├── Main_loop.py
│   ├── Obstacles.py
│   └── ...
├── RRTStar/
│   ├── Environment.py
│   ├── KinematicBycicleModelRRT.py
│   ├── main.py
│   ├── RRTStar.py
│   ├── utils.py
│   ├── ...
└── README.md
```

## Dependencies

This project uses the following libraries.
- Python 3.8+
- PyBullet
- ACADOS
- CasADi
- NumPy
- SciPy

Install them using

```bash
pip install pybullet numpy scipy casadi
```

ACADOS requires a separate installation involving a C compiler and CMake. Use the [documentation](https://docs.acados.org/installation/index.html) for this. It is recommended to run it on LINUX.

## Running the Simulation
For RRT*, navigate to the RRTStar directory and run the script ```main.py```.

For MPC, navigate to the MPC directory and run the script ```Main_loop.py```.
