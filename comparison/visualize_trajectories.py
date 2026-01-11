#!/usr/bin/env python3
"""
Visualizes the trajectories of RRT*, Pure MPC, and MPC+RRT* from the latest logs.
Generates a comparative plot 'comparison_plot.png'.
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Import shared config for obstacles
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_config import STATIC_OBSTACLES, MAP_SIZE, START_POS, GOAL_POS

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

def get_latest_log(pattern):
    search_pattern = os.path.join(LOG_DIR, pattern)
    files = glob.glob(search_pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def load_trajectory(log_file, method_type):
    with open(log_file, 'r') as f:
        data = json.load(f)
        
    x = []
    y = []
    
    if method_type == "RRT":
        # RRT log has 'path' which is list of objects or list of tuples
        # Check structure
        path = data.get('path', [])
        # RRT path is typically [x, y] or node dicts
        for p in path:
            if isinstance(p, dict):
                x.append(p['x'])
                y.append(p['y'])
            elif isinstance(p, list):
                x.append(p[0])
                y.append(p[1])
    else:
        # MPC / MPC+RRT logs have 'trajectory' list
        traj = data.get('trajectory', [])
        for step in traj:
            # step can be list [x, y, v, psi] or dict
            if isinstance(step, list):
                 x.append(step[0])
                 y.append(step[1])
            elif 'state' in step:
                 x.append(step['state']['x'])
                 y.append(step['state']['y'])
            else:
                 x.append(step.get('x', 0))
                 y.append(step.get('y', 0))
                 
    return x, y

def plot_environment(ax):
    # Plot Map Boundary
    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    ax.set_aspect('equal')
    
    # Plot Obstacles
    for obs in STATIC_OBSTACLES:
        circle = patches.Circle(obs['position'], obs['radius'], color='k', alpha=0.5)
        ax.add_patch(circle)
        
    # Plot Start/Goal
    ax.plot(START_POS[0], START_POS[1], 'go', markersize=10, label='Start')
    ax.plot(GOAL_POS[0], GOAL_POS[1], 'ro', markersize=10, label='Goal')
    
    ax.grid(True, alpha=0.3)
    ax.set_title("Trajectory Comparison: RRT* vs MPC vs Integrated")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

def main():
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_environment(ax)
    
    configs = [
        ("Pure RRT*", "rrt_comparison_*.json", "blue", "--"),
        ("Pure MPC", "mpc_comparison_*.json", "red", ":"),
        ("MPC + RRT*", "mpc_rrt_comparison_*.json", "green", "-")
    ]
    
    for name, pattern, color, style in configs:
        log_file = get_latest_log(pattern)
        if not log_file:
            print(f"Skipping {name}: No log found")
            continue
            
        print(f"Plotting {name} from {os.path.basename(log_file)}")
        try:
            x, y = load_trajectory(log_file, "RRT" if "RRT" in name and "MPC" not in name else "MPC")
            if x and y:
                ax.plot(x, y, color=color, linestyle=style, linewidth=2, label=name)
                # Plot end point
                ax.plot(x[-1], y[-1], marker='x', color=color, markersize=8)
        except Exception as e:
            print(f"Error reading {name}: {e}")
            
    ax.legend()
    
    output_path = os.path.join(os.path.dirname(__file__), "comparison_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
