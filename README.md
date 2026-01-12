# PDM_27 - Path Planning and Control

Autonomous vehicle path planning using RRT*, MPC, and hybrid RRT*+MPC approaches.

## Environment Requirements

### System Requirements
- Python 3.10+
- Linux (tested on Ubuntu 22.04。5 LTS jammy)
- ACADOS solver (installed separately)

### Python Dependencies

```bash
pip install numpy pybullet casadi scipy matplotlib pandas
```

### ACADOS Installation

ACADOS must be installed from source:

```bash
# Clone and build ACADOS
git clone https://github.com/acados/acados.git ~/acados
cd ~/acados
git submodule update --recursive --init
mkdir -p build && cd build
cmake .. -DACADOS_WITH_QPOASES=ON
make install -j4

# Install Python interface
pip install -e ~/acados/interfaces/acados_template
```

Set environment variables (add to `.bashrc`):

```bash
export ACADOS_SOURCE_DIR=~/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/acados/lib
```

## Quick Start

```bash
# Run RRT*+MPC simulation (GUI)
python MPC/Main_loop_MPC_RRT.py

# Run Pure MPC simulation (GUI)
python MPC/Main_loop.py

# Run comprehensive benchmark (headless)
python run_comprehensive_benchmark.py
```

## File Structure

```
PDM_27/
├── MPC/                          # MPC controller and simulations
│   ├── Main_loop.py              # Pure MPC simulation loop
│   ├── Main_loop_MPC_RRT.py      # RRT*+MPC simulation loop
│   ├── MPC_core.py               # ACADOS MPC solver implementation
│   ├── MPC_core_lighthouse_nav.py# MPC ‘lighthouse’ mode (no reference path)
│   ├── MPC_RRT.py                # RRT*+MPC hybrid controller
│   ├── Config.py                 # Vehicle and simulation parameters
│   ├── Obstacles.py              # Obstacle generation, save/load
│   ├── Environment.py            # PyBullet environment for MPC
│   ├── Trajectory_generator_obs.py # Obstacle trajectory generation
│   ├── analyze_log.py            # Script to analyze simulation logs
│   ├── check_log.py              # Script to check logs
│   ├── debug_mpc.py              # Debugging script for MPC
│   ├── quick_analysis.py         # Quick analysis script
│   ├── test_dynamics.py          # Script for testing vehicle dynamics
│   └── saved_obstacles.json      # Saved obstacle config (shared)
│
├── RRTStar/                      # RRT* path planner
│   ├── RRTStar.py                # RRT* algorithm implementation
│   ├── main.py                   # Standalone RRT* runner
│   ├── KinematicBicycleModelRRT.py  # Vehicle kinematics for RRT
│   ├── Environment.py            # PyBullet environment for RRT
│   └── utils.py                  # Utility functions
│
├── Environment/                  # Shared environment utilities
│   ├── environment.py            # Base environment class
│   └── maze_layouts.py           # Predefined obstacle layouts
│
├── comparison/                   # Single-run comparison scripts
│   ├── rrt_comparison.py         # Pure RRT* benchmark
│   ├── mpc_comparison.py         # Pure MPC benchmark
│   ├── mpc_rrt_comparison.py     # RRT*+MPC benchmark
│   ├── analyze_comparison.py     # Analyze benchmark results
│   └── visualize_trajectories.py # Plot trajectory comparisons
│
├── logs/                         # Simulation logs (JSON)
├── benchmark_plots/              # Plots from benchmarks
│
├── run_benchmark.sh              # Runs a series of comparison scripts
├── run_comprehensive_benchmark.py # Main benchmark runner
├── generate_benchmark_plots.py   # Generate report plots from benchmark data
└── README.md                     # This file
```

## Core Components

### MPC/Config.py
Vehicle and simulation parameters:
- `vehicle_length`, `vehicle_width`: Physical dimensions
- `max_speed`, `max_steering_angle`: Control limits
- `controller_dt`: Control loop frequency (0.05s = 20Hz)
- `mpc_dt`, `mpc_horizon`: MPC prediction settings

### MPC/MPC_core.py
ACADOS-based MPC solver:
- Nonlinear least squares cost function
- Softplus barrier for obstacle avoidance
- SQP_RTI solver for real-time performance
- Deadlock detection and recovery

### MPC/MPC_RRT.py
Hybrid RRT*+MPC controller:
- RRT* for global path planning
- MPC for local trajectory tracking
- Waypoint-following with lookahead
- Dynamic obstacle handling

### RRTStar/RRTStar.py
Sampling-based path planner:
- Goal-biased random sampling
- Tree rewiring for path optimization
- Kinematic feasibility checking
- Cubic spline path smoothing

### MPC/Obstacles.py
Obstacle management:
- Random obstacle generation
- `save_obstacles()`: Persist to JSON
- `load_obstacles()`: Load from JSON
- Enables map sharing between simulations

## Running Simulations

### Pure MPC (with GUI)
```bash
python MPC/Main_loop.py
```

### RRT*+MPC (with GUI)
```bash
python MPC/Main_loop_MPC_RRT.py
```

### Headless Mode
Add the `--headless` flag to run any simulation without the GUI:
```bash
python MPC/Main_loop.py --headless
python MPC/Main_loop_MPC_RRT.py --headless
```

### Map Sharing
1. Run `python MPC/Main_loop_MPC_RRT.py` first - it generates a random map and saves it to `MPC/saved_obstacles.json`.
2. Run `python MPC/Main_loop.py` - it will load the same map for a direct comparison.

## Benchmarking

### Comprehensive Benchmark (headless)
This script runs a comprehensive benchmark for multiple algorithms and scenarios.
```bash
python run_comprehensive_benchmark.py
```
Check the script for configuration options. Results and plots are saved in the `benchmark_plots/` directory.

### Single Comparisons (headless by default)
These scripts run a single trial for each specified algorithm.
```bash
cd comparison
python rrt_comparison.py
python mpc_comparison.py
python mpc_rrt_comparison.py
```

## Output Files

### Logs (`logs/`)
JSON files with detailed simulation data for each run, including trajectory, controls, and solver performance.

### Plots (`benchmark_plots/`)
- Contains plots summarizing the results of the comprehensive benchmark, such as success rate, collision rate, and computation time.
- `summary_table.csv`: Tabular results of the benchmark.


