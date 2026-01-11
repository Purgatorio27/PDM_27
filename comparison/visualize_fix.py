#!/usr/bin/env python3
"""
Visualizes the fix: Compares the failed Pure MPC run with the fixed MPC+RRT* run.
Generates a plot in 'comparison/plots/' directory.
Includes dynamic obstacles plotting.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import os
import sys
import numpy as np
from collections import defaultdict

# Add parent directory to path to allow importing shared_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from shared_config import STATIC_OBSTACLES, MAP_SIZE, START_POS, GOAL_POS, VEHICLE_LENGTH, VEHICLE_WIDTH
except ImportError:
    print("Warning: Could not import shared_config. using defaults.")
    STATIC_OBSTACLES = []
    MAP_SIZE = 30.0
    START_POS = (2.0, 2.0)
    GOAL_POS = (26.0, 26.0)
    VEHICLE_LENGTH = 1.5
    VEHICLE_WIDTH = 1.0

# Define output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

# Specific logs identified by the agent
MPC_LOG_FILE = "mpc_comparison_2026-01-11_22-01-26.json"
RRT_LOG_FILE = "dynamic_mpc_rrt_2026-01-11_22-22-43.json"

def load_log(filename):
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: Log file not found: {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def plot_car(ax, x, y, yaw, color='blue', alpha=0.5):
    """Draws a car rectangle rotated by yaw."""
    # Create rectangle centered at (0,0) then rotate and translate
    rect = patches.Rectangle(
        (-VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2), 
        VEHICLE_LENGTH, VEHICLE_WIDTH, 
        linewidth=1, edgecolor='black', facecolor=color, alpha=alpha
    )
    
    t = transforms.Affine2D().rotate(yaw) + transforms.Affine2D().translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


def plot_dynamic_obstacles(ax, log_data, max_steps=None):
    """
    Plots the trails of dynamic obstacles.
    """
    if 'trajectory' not in log_data:
        return

    traj = log_data['trajectory']
    if max_steps:
        traj = traj[:max_steps]

    # Organize by Obstacle ID
    obs_paths = defaultdict(list)
    
    for step in traj:
        if 'dynamic_obstacles' in step:
            for obs in step['dynamic_obstacles']:
                obs_id = obs.get('id', 0)
                obs_paths[obs_id].append((obs['x'], obs['y'], obs.get('radius', 1.0)))
    
    # Define colors for multiple obstacles
    cmap = plt.get_cmap('tab10')
    
    for obs_id, path in obs_paths.items():
        if not path:
            continue
            
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        radii = [p[2] for p in path]
        
        # Plot Path Trail
        color = cmap(obs_id % 10)
        ax.plot(xs, ys, color=color, linewidth=1, linestyle=':', alpha=0.6)
        
        # Plot Start Position (faint)
        circle_start = plt.Circle((xs[0], ys[0]), radii[0], color=color, alpha=0.1)
        ax.add_patch(circle_start)
        
        # Plot End Position (solid)
        circle_end = plt.Circle((xs[-1], ys[-1]), radii[-1], color=color, alpha=0.4, label=f'Dyn Obs {obs_id}' if obs_id == 0 else "")
        ax.add_patch(circle_end)
        ax.text(xs[-1], ys[-1], str(obs_id), color='white', ha='center', va='center', fontsize=8, fontweight='bold')


def plot_trajectory(ax, log_data, label, color, linestyle='-'):
    if 'trajectory' not in log_data:
        return
    
    traj = log_data['trajectory']
    xs = [step['x'] for step in traj]
    ys = [step['y'] for step in traj]
    yaws = [step.get('psi', 0) for step in traj] # Assuming 'psi' is yaw, might be 'yaw'
    if not yaws and len(traj) > 0 and 'yaw' in traj[0]:
         yaws = [step.get('yaw', 0) for step in traj]

    # Plot Line
    ax.plot(xs, ys, label=label, color=color, linewidth=2, linestyle=linestyle)
    
    # Plot Start and End Car
    if len(xs) > 0:
        plot_car(ax, xs[0], ys[0], yaws[0] if yaws else 0, color=color, alpha=0.3) # Start
        plot_car(ax, xs[-1], ys[-1], yaws[-1] if yaws else 0, color=color, alpha=0.8) # End

def main():
    mpc_data = load_log(MPC_LOG_FILE)
    rrt_data = load_log(RRT_LOG_FILE)

    if not mpc_data or not rrt_data:
        return

    # --- Print Summary ---
    print(f"{'Metric':<30} | {'Pure MPC (Previous)':<20} | {'MPC + RRT* (Fixed)':<20}")
    print("-" * 75)
    
    mpc_col = mpc_data.get('summary', {}).get('collision_count', 'N/A')
    rrt_col = rrt_data.get('collisions', rrt_data.get('summary', {}).get('collision_count', 'N/A'))
    
    mpc_succ = mpc_data.get('summary', {}).get('goal_reached', False)
    # RRT log check
    if 'summary' in rrt_data:
        rrt_succ = rrt_data['summary'].get('goal_reached', False)
    else:
        rrt_succ = (rrt_col == 0)

    print(f"{'Collisions':<30} | {mpc_col:<20} | {rrt_col:<20}")
    print(f"{'Success':<30} | {str(mpc_succ):<20} | {str(rrt_succ):<20}")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Plot Map Boundary
    ax.plot([0, MAP_SIZE, MAP_SIZE, 0, 0], [0, 0, MAP_SIZE, MAP_SIZE, 0], 'k-', linewidth=3, label="Boundary")
    
    # 2. Plot Static Obstacles
    for i, obs in enumerate(STATIC_OBSTACLES):
        circle = patches.Circle(obs['position'], obs['radius'], color='#404040', alpha=0.6, label="Static Obs" if i==0 else "")
        ax.add_patch(circle)
        
    # 3. Plot Dynamic Obstacles (From the SUCCESSFUL run to show how we avoided them)
    # We use the RRT data for dynamic obstacles as it's the longer, successful run
    plot_dynamic_obstacles(ax, rrt_data)
    
    # 4. Plot Start and Goal Icons
    ax.plot(START_POS[0], START_POS[1], marker='o', color='green', markersize=12, label='Start', linestyle='None')
    ax.plot(GOAL_POS[0], GOAL_POS[1], marker='*', color='#FFD700', markersize=18, markeredgecolor='black', label='Goal', linestyle='None') # Gold star

    # 5. Plot Trajectories
    plot_trajectory(ax, mpc_data, f"Pure MPC (Collisions: {mpc_col})", 'red', '--')
    plot_trajectory(ax, rrt_data, f"MPC + RRT* (Fixed, Collisions: {rrt_col})", 'blue', '-')

    # Improve Styling
    ax.set_title("Fix Verification: Pure MPC vs. MPC + RRT* with Dynamic Obstacles", fontsize=16)
    ax.set_xlabel("X Position [m]", fontsize=12)
    ax.set_ylabel("Y Position [m]", fontsize=12)
    
    # Legend - handle duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', framealpha=0.9)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust Plot Limits with padding
    ax.set_aspect('equal')
    ax.set_xlim(-2, MAP_SIZE + 2)
    ax.set_ylim(-2, MAP_SIZE + 2)

    output_path = os.path.join(OUTPUT_DIR, 'fix_verification_detailed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDetailed plot saved to: {output_path}")

if __name__ == "__main__":
    main()
