
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import os
import sys
import numpy as np
from collections import defaultdict

SUMMARY_FILE = 'benchmark_summary.json'
LOG_DIR = 'logs'
OUTPUT_DIR = 'benchmark_plots/trajectories'
VEHICLE_LENGTH = 1.5
VEHICLE_WIDTH = 1.0

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_log(filename):
    if not os.path.isabs(filename):
        # Check if it has logs/ prefix
        if not filename.startswith(LOG_DIR):
             filename = os.path.join(LOG_DIR, os.path.basename(filename))
        else:
             # it has logs/ but is relative
             if not os.path.exists(filename):
                 # Try relative to cwd
                 pass
    
    if not os.path.exists(filename):
        print(f"Error: Log file not found: {filename}")
        return None
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

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

def analyze_collisions(trajectory, static_obstacles, dynamic_obstacles_by_step, radius=1.0):
    collision_points = []
    
    for i, step in enumerate(trajectory):
        x, y = step['x'], step['y']
        
        # Check static
        for obs in static_obstacles:
            ox, oy = obs['position'] if 'position' in obs else (obs['x'], obs['y'])
            r = obs['radius']
            if np.hypot(x-ox, y-oy) < (r + radius):
                collision_points.append((x, y))
                continue 
        
        # Check dynamic
        if dynamic_obstacles_by_step and i < len(dynamic_obstacles_by_step):
            dyn_obs = dynamic_obstacles_by_step[i]
            for dobs in dyn_obs:
                if np.hypot(x-dobs['x'], y-dobs['y']) < (dobs['radius'] + radius):
                     collision_points.append((x, y))
    
    return collision_points

def plot_single_run(run_info):
    log_file = run_info['log_file']
    data = load_log(log_file)
    if not data:
        return

    metadata = data.get('metadata', {})
    map_size = metadata.get('map_size', 30.0)
    static_obstacles = metadata.get('static_obstacles', [])
    start = metadata.get('start', [2.0, 2.0])
    goal = metadata.get('goal', [26.0, 26.0])
    
    trajectory = data.get('trajectory', [])
    if not trajectory:
        print(f"No trajectory in {log_file}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect('equal')
    
    # Plot Static Obstacles
    for obs in static_obstacles:
        # Check format
        if 'position' in obs:
            pos = obs['position']
            r = obs['radius']
        else:
            pos = [obs['x'], obs['y']]
            r = obs['radius']
            
        circle = patches.Circle(pos, r, color='gray', alpha=0.5)
        ax.add_patch(circle)
        
    # Plot Dynamic Obstacles (Trails)
    dynamic_obstacles_by_step = []
    if run_info['scenario'] == 'dynamic':
        # Collect dynamic obstacles per step
        obs_paths = defaultdict(list)
        for step_idx, step in enumerate(trajectory):
            step_obs = step.get('dynamic_obstacles', [])
            dynamic_obstacles_by_step.append(step_obs)
            
            for obs in step_obs:
                oid = obs.get('id', 0) # Assumes ID availability or use index if consistent
                obs_paths[oid].append((obs['x'], obs['y'], obs.get('radius', 1.0)))
        
        # Plot trails
        cmap = plt.get_cmap('tab10')
        for oid, path in obs_paths.items():
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            # Trail
            ax.plot(xs, ys, color=cmap(oid%10), alpha=0.2, linewidth=1, linestyle='--')
            # Final position
            last = path[-1]
            circle = patches.Circle((last[0], last[1]), last[2], color=cmap(oid%10), alpha=0.3, fill=False)
            ax.add_patch(circle)
    
    # Plot Vehicle Trajectory
    xs = [t['x'] for t in trajectory]
    ys = [t['y'] for t in trajectory]
    
    algo_color = {
        'RRT': 'blue',
        'MPC': 'orange',
        'MPC_RRT': 'green'
    }.get(run_info['algorithm'], 'black')
    
    ax.plot(xs, ys, color=algo_color, linewidth=2, label='Trajectory')
    
    # Plot Start/Goal
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'rx', markersize=10, label='Goal')
    
    # Plot Vehicle at start and end
    # Need orientation. If missing, estimate from diff
    if len(xs) > 1:
        # Start
        yaw_start = np.arctan2(ys[1]-ys[0], xs[1]-xs[0])
        plot_car(ax, xs[0], ys[0], yaw_start, color='green', alpha=0.3)
        
        # End
        yaw_end = np.arctan2(ys[-1]-ys[-2], xs[-1]-xs[-2])
        plot_car(ax, xs[-1], ys[-1], yaw_end, color=algo_color, alpha=0.8)

    # Collisions
    # Get collisions from simple analysis or log?
    # Log has "collisions" count but not locations usually.
    # Let's re-calculate approx collision locations
    collisions = analyze_collisions(trajectory, static_obstacles, 
                                    dynamic_obstacles_by_step if run_info['scenario'] == 'dynamic' else [],
                                    radius=metadata.get('vehicle_radius', 0.9))
    
    if collisions:
        cx, cy = zip(*collisions)
        ax.plot(cx, cy, 'rx', markersize=8, markeredgewidth=2, label='Collision')

    title = f"{run_info['algorithm']} on {run_info['map_size']}x{run_info['map_size']} ({run_info['scenario']})"
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    filename = f"{run_info['map_size']}_{run_info['scenario']}_{run_info['algorithm']}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved {filepath}")

def main():
    if not os.path.exists(SUMMARY_FILE):
        print("Summary file not found.")
        return
        
    with open(SUMMARY_FILE, 'r') as f:
        runs = json.load(f)
        
    for run in runs:
        plot_single_run(run)

if __name__ == "__main__":
    main()
