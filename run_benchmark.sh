#!/bin/bash
set -e  # Exit on error

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Source the environment setup
source "$DIR/activate_env.sh"

echo "========================================="
echo "Starting Benchmark Suite"
echo "========================================="

# 1. Run Pure RRT* (Headless)
echo "[1/3] Running Pure RRT* Simulation..."
python3 "$DIR/comparison/rrt_comparison.py" --headless

# 2. Run Pure MPC (Headless)
echo "[2/3] Running Pure MPC Simulation..."
python3 "$DIR/comparison/mpc_comparison.py" --headless

# 3. Run MPC + RRT* (Headless)
echo "[3/3] Running MPC + RRT* Simulation..."
python3 "$DIR/comparison/mpc_rrt_comparison.py" --headless

# 4. Analyze Results
echo "[4/4] Analyzing Results..."
python3 "$DIR/comparison/analyze_comparison.py"

echo "========================================="
echo "Benchmark Complete!"
echo "Check comparison/plots/ for visualizations."
echo "========================================="
