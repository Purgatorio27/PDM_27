#!/bin/bash

# Setup environment
source activate_env.sh

echo "========================================="
echo "Starting Dynamic Benchmark (30x30)"
echo "========================================="

# 1. Run Pure MPC with Dynamic Obstacles
echo "[1/2] Running Dynamic Pure MPC..."
python3 comparison/run_dynamic_mpc.py --headless

# 2. Run MPC + RRT* with Dynamic Obstacles
echo "[2/2] Running Dynamic MPC + RRT*..."
python3 comparison/run_dynamic_mpc_rrt.py --headless

# 3. Analyze
echo "[3/3] Analyzing Dynamic Results..."
python3 comparison/analyze_dynamic_results.py

echo "Done."
