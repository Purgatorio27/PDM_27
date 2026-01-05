import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import glob


ANALYZE_ALL = True


def load_latest_log():
    """Load the most recent simulation log"""
    log_dir = os.path.join(os.getcwd(), 'logs')
    log_files = glob.glob(os.path.join(log_dir, 'rrt_log_*.json'))
    
    if not log_files:
        print("No log files found in logs/ directory")
        return None
    
    latest_file = max(log_files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def plot_rrt_results(log_data):
    """Plot the RRT* planned path and the executed trajectory"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    goal = log_data['metadata']['goal_pos']
    start = log_data['metadata']['start_pose']
    
    if 'planned_rrt_path' in log_data and log_data['planned_rrt_path']:
        px = [p['x'] for p in log_data['planned_rrt_path']]
        py = [p['y'] for p in log_data['planned_rrt_path']]
        ax.plot(px, py, 'b--', alpha=0.6, linewidth=2, label='Planned RRT* Path')

    traj = log_data['executed_trajectory']
    tx = [t['x'] for t in traj]
    ty = [t['y'] for t in traj]
    
    if 'v' in traj[0]:
        v = [t['v'] for t in traj]

        points = np.array([tx, ty]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(min(v), max(v))

        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=4, label='Executed Trajectory')
        lc.set_array(np.array(v))

        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label='Velocity (m/s)')
    else:
        ax.plot(tx, ty, 'r-', linewidth=3, label='Executed Trajectory')

    ax.plot(start[0], start[1], 'bo', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'g*', markersize=20, label='Goal')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('RRT* Path Planning and Execution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return fig

def plot_velocity_profile(log_data):
    """Analyze velocity and heading over the steps"""
    traj = log_data['executed_trajectory']
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
        ax2.set_ylabel('Heading (degrees)')
        ax2.set_xlabel('Trajectory Step')
        ax2.set_title('Vehicle Orientation (Yaw)')
        ax2.grid(True)

    plt.tight_layout()
    return fig

def print_rrt_summary(log_data):
    """Print summary statistics"""
    meta = log_data['metadata']
    traj = log_data['executed_trajectory']
    
    print("\n" + "="*60)
    print("RRT* SIMULATION LOG SUMMARY")
    print("="*60)

    print(f"Start Pose: {meta['start_pose']}")
    print(f"Goal Position: {meta['goal_pos']}")
    
    comp_time = meta.get('computation_time_sec', 0)
    trav_time = meta.get('travel_time_sec', 0)
    
    print(f"Computation Time: {comp_time:.4f} seconds")
    print(f"Travel Time: {trav_time:.2f} seconds")
    
    if len(traj) > 1:
        dist = 0
        for i in range(len(traj)-1):
            dist += np.hypot(traj[i+1]['x'] - traj[i]['x'], traj[i+1]['y'] - traj[i]['y'])
        print(f"Total Distance: {dist:.2f} meters")
        print(f"Average Speed: {dist/trav_time:.2f} m/s" if trav_time > 0 else "N/A")
    
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