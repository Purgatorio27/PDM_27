import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import glob


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)


def load_latest_log():
    """Search through nested folders for the most recent simulation log"""
    log_pattern = os.path.join(root_dir, 'logs', '*', '*', '*.json')
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print(f"No log files found in {os.path.join(root_dir, 'logs')}")
        return None
    
    # Sort by creation time to find the newest
    latest_file = max(log_files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def plot_rrt_results(log_data):
    """Plot the RRT* planned path and the executed trajectory"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    goal = log_data['metadata']['goal_pos']
    start = log_data['metadata']['start_pose']
    
    # Plot Trajectory
    traj = log_data['executed_trajectory']
    tx = [t['x'] for t in traj]
    ty = [t['y'] for t in traj]
    
    # Use a color map for velocity if 'v' exists
    if len(traj) > 0 and 'v' in traj[0]:
        v = [t['v'] for t in traj]
        points = np.array([tx, ty]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Avoid error if velocity is constant
        v_min, v_max = min(v), max(v)
        if v_min == v_max: v_max += 0.1
        
        norm = plt.Normalize(v_min, v_max)
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=4)
        lc.set_array(np.array(v))
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label='Velocity (m/s)')
    else:
        ax.plot(tx, ty, 'r-', linewidth=3, label='Executed Trajectory')

    ax.plot(start[0], start[1], 'bo', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'g*', markersize=20, label='Goal')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    difficulty = log_data['metadata']['parameters'].get('difficulty', 'Unknown')
    ax.set_title(f'RRT* Path Analysis: {difficulty} Maze')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return fig

def plot_velocity_profile(log_data):
    """Analyze velocity and heading over the steps"""
    traj = log_data['executed_trajectory']
    if not traj: return None

    steps = [t['step'] for t in traj]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    if 'v' in traj[0]:
        v = [t['v'] for t in traj]
        ax1.plot(steps, v, 'g-', linewidth=2)
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title('Velocity Profile')
        ax1.grid(True)

    if 'yaw' in traj[0]:
        yaw = [np.degrees(t['yaw']) for t in traj]
        ax2.plot(steps, yaw, 'b-', linewidth=2)
        ax2.set_ylabel('Heading (deg)')
        ax2.set_xlabel('Trajectory Step')
        ax2.set_title('Vehicle Orientation (Yaw)')
        ax2.grid(True)

    plt.tight_layout()
    return fig

def print_rrt_summary(log_data):
    """Print summary statistics using updated keys"""
    meta = log_data['metadata']
    
    print("\n" + "="*60)
    print(f"RRT* SUMMARY: {meta['parameters'].get('difficulty', 'Unknown')} Maze")
    print("="*60)

    print(f"Start Pose: {meta['start_pose']}")
    print(f"Goal Pos:   {meta['goal_pos']}")
    
    comp_time = meta.get('computation_time', 0)
    trav_time = meta.get('travel_time', 0)
    dist = meta.get('total_distance', 0)
    
    print(f"Planning Time:   {comp_time:.4f} sec")
    print(f"Execution Time:  {trav_time:.2f} sec")
    print(f"Total Distance:  {dist:.2f} meters")
    
    if trav_time > 0:
        print(f"Average Speed:   {dist/trav_time:.2f} m/s")
    
    print("="*60)

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            log_data = json.load(f)
    else:
        log_data = load_latest_log()
    
    if log_data:
        print_rrt_summary(log_data)
        plot_rrt_results(log_data)
        plot_velocity_profile(log_data)
        plt.show()

if __name__ == "__main__":
    main()