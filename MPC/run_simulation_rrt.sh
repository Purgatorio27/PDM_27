#!/bin/bash
# MPC Simulation Runner Script
# Automatically cleans up old solver cache and runs the simulation

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MPC_DIR="$SCRIPT_DIR"

echo "========================================"
echo "MPC Simulation Runner"
echo "========================================"

# Activate environment
if [ -f "$PROJECT_DIR/activate_env.sh" ]; then
    echo "[1/4] Activating environment..."
    source "$PROJECT_DIR/activate_env.sh"
    # Ensure Acados vars are exported
    export ACADOS_SOURCE_DIR=/home/yanghongyi/acados
    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
else
    echo "Warning: activate_env.sh not found, continuing without it"
    export ACADOS_SOURCE_DIR=/home/yanghongyi/acados
    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
fi

# Clean up old solver files
echo "[2/4] Cleaning up old solver cache..."
rm -rf "$MPC_DIR/c_generated_code" 2>/dev/null || true
rm -f "$MPC_DIR/vehicle_mpc.json" 2>/dev/null || true
rm -rf "$MPC_DIR/__pycache__" 2>/dev/null || true
echo "    - c_generated_code: removed"
echo "    - vehicle_mpc.json: removed"
echo "    - __pycache__: removed"

# Change to MPC directory
cd "$MPC_DIR"

# Run simulation
echo "[3/4] Starting MPC+RRT simulation..."
echo "----------------------------------------"
/home/yanghongyi/PDM_27/venv/bin/python Main_loop_MPC_RRT.py

# Analyze results
echo ""
echo "[4/4] Analyzing results..."
echo "----------------------------------------"
if [ -f "check_log.py" ]; then
    python3 check_log.py
fi

echo ""
echo "========================================"
echo "Simulation complete!"
echo "========================================"
