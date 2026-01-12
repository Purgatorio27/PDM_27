# PDM_27 - Path Planning and Control

Autonomous vehicle path planning using RRT*, MPC, and hybrid RRT*+MPC approaches.

## Environment Requirements

### System Requirements
- Python 3.10+
- Linux (tested on WSL2)
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
# Activate virtual environment
source venv/bin/activate

# Run Pure MPC simulation (GUI)
cd MPC && bash run_simulation.sh

# Run RRT*+MPC simulation (GUI)
cd MPC && bash run_simulation_rrt.sh

# Run mass benchmark (headless, 100 trials each)
cd mass_bench && python run_mass_benchmark.py
```

## File Structure

```
PDM_27/
├── MPC/                          # MPC controller and simulations
│   ├── Config.py                 # Vehicle and simulation parameters
│   ├── MPC_core.py               # ACADOS MPC solver implementation
│   ├── MPC_RRT.py                # RRT*+MPC hybrid controller
│   ├── Main_loop.py              # Pure MPC simulation loop
│   ├── Main_loop_MPC_RRT.py      # RRT*+MPC simulation loop
│   ├── Obstacles.py              # Obstacle generation, save/load
│   ├── Environment.py            # PyBullet environment setup
│   ├── run_simulation.sh         # Run pure MPC
│   ├── run_simulation_rrt.sh     # Run RRT*+MPC
│   ├── saved_obstacles.json      # Saved obstacle config (shared)
│   └── c_generated_code/         # ACADOS generated C code
│
├── RRTStar/                      # RRT* path planner
│   ├── RRTStar.py                # RRT* algorithm implementation
│   ├── KinematicBicycleModelRRT.py  # Vehicle kinematics for RRT
│   ├── Environment.py            # PyBullet environment for RRT
│   ├── main.py                   # Standalone RRT* runner
│   └── utils.py                  # Utility functions
│
├── comparison/                   # Single-run comparison scripts
│   ├── rrt_comparison.py         # Pure RRT* benchmark
│   ├── mpc_comparison.py         # Pure MPC benchmark
│   ├── mpc_rrt_comparison.py     # RRT*+MPC benchmark
│   ├── run_dynamic_mpc.py        # MPC with dynamic obstacles
│   ├── run_dynamic_mpc_rrt.py    # RRT*+MPC with dynamic obstacles
│   ├── analyze_comparison.py     # Analyze benchmark results
│   └── visualize_trajectories.py # Plot trajectory comparisons
│
├── mass_bench/                   # Mass benchmarking (100 trials)
│   ├── run_mass_benchmark.py     # Main benchmark runner
│   ├── generate_plots.py         # Generate report plots
│   ├── run_benchmark.sh          # Bash runner script
│   ├── results/                  # CSV/JSON results
│   └── plots/                    # Generated plots
│
├── Environment/                  # Shared environment utilities
│   ├── environment.py            # Base environment class
│   └── maze_layouts.py           # Predefined obstacle layouts
│
├── logs/                         # Simulation logs (JSON)
├── c_generated_code/             # Root-level ACADOS code
├── venv/                         # Python virtual environment
├── ALGO.md                       # Algorithm documentation
├── PROJECT.md                    # This file
└── requirements.txt              # Python dependencies
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
cd MPC
bash run_simulation.sh
```

### RRT*+MPC (with GUI)
```bash
cd MPC
bash run_simulation_rrt.sh
```

### Headless Mode
Add `--headless` flag in shell scripts or set `gui=False` in code.

### Map Sharing
1. Run `run_simulation_rrt.sh` first - generates random map and saves to `saved_obstacles.json`
2. Run `run_simulation.sh` - loads the same map for comparison

## Benchmarking

### Mass Benchmark (100 trials)
```bash
cd mass_bench
python run_mass_benchmark.py
```

Generates:
- `results/benchmark_results_final.csv`: Raw trial data
- `results/summary_statistics.json`: Aggregated metrics
- `plots/`: Visualization (success rate, path length, efficiency, computation time)

### Single Comparisons
```bash
cd comparison
python rrt_comparison.py      # Pure RRT*
python mpc_comparison.py      # Pure MPC
python mpc_rrt_comparison.py  # RRT*+MPC
```

## Output Files

### Logs (`logs/`)
JSON files with simulation data:
- `trajectory`: Vehicle path points
- `controls`: Applied steering and acceleration
- `summary`: Success, path length, computation time
- `planning_info`: RRT* path data (if applicable)

### Plots (`mass_bench/plots/`)
- `success_rate.png`: Algorithm success comparison
- `path_length_distribution.png`: Path length histograms
- `path_efficiency_distribution.png`: Efficiency comparison
- `computation_time.png`: Runtime comparison
- `combined_summary.png`: Multi-panel summary
- `summary_table.csv`: Tabular results

## Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Start position | (2, 2) | Config.py |
| Goal position | (36, 36) | Config.py |
| Map size | 45 m | MPC_RRT.py |
| Max speed | 4.0 m/s | Config.py |
| MPC horizon | 60 steps | Config.py |
| MPC dt | 0.25 s | Config.py |
| Controller dt | 0.05 s | Config.py |
| Goal tolerance | 1.5 m | MPC_core.py |
