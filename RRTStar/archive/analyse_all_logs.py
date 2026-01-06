"""
RRT* Simulation Log Analyzer and Visualizer
Supports Single-run visualization and Batch-averaging mode with Distribution Plots.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import glob


ANALYZE_ALL = True
LOG_DIR = os.path.join(os.getcwd(), 'logs')


def load_latest_log():
    """Load the most recent simulation log."""
    log_files = glob.glob(os.path.join(LOG_DIR, 'rrt_log_*.json'))
    if not log_files:
        print(f"No log files found in {LOG_DIR}")
        return None
    latest_file = max(log_files, key=os.path.getctime)
    print(f"Loading latest: {latest_file}")
    with open(latest_file, 'r') as f:
        return json.load(f)

def plot_batch_metrics(comp_times, trav_times, distances):
    """Generates distribution plots for batch analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(comp_times, bins=15, color='skyblue', edgecolor='black')
    axes[0].set_title('RRT* Solve Time Distribution')
    axes[0].set_xlabel('Seconds')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].boxplot(trav_times, vert=True, patch_artist=True, 
                    boxprops=dict(facecolor='lightgreen'))
    axes[1].set_title('Travel Time Consistency')
    axes[1].set_ylabel('Seconds')
    axes[1].set_xticklabels(['All Runs'])
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].scatter(distances, comp_times, color='coral', alpha=0.6)
    axes[2].set_title('Path Length vs. Solve Time')
    axes[2].set_xlabel('Path Distance (m)')
    axes[2].set_ylabel('Solve Time (s)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def analyze_all_logs():
    """Averages performance metrics across all log files and generates batch plots."""
    log_files = glob.glob(os.path.join(LOG_DIR, 'rrt_log_*.json'))

    if not log_files:
        print("No log files found to average.")
        return

    comp_times = []
    trav_times = []
    distances = []
    avg_speeds = []
    success_count = 0

    for file_path in log_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                meta = data.get('metadata', {})
                traj = data.get('executed_trajectory', [])

                c_time = meta.get('computation_time_sec', 0)
                t_time = meta.get('travel_time_sec', 0)
                
                if len(traj) > 0:
                    success_count += 1
                    comp_times.append(c_time)
                    trav_times.append(t_time)

                    d = sum(np.hypot(traj[i+1]['x'] - traj[i]['x'], 
                                     traj[i+1]['y'] - traj[i]['y']) 
                            for i in range(len(traj)-1))
                    distances.append(d)
                    if t_time > 0:
                        avg_speeds.append(d / t_time)
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    if not comp_times:
        print("No valid successful runs found in logs.")
        return

    print("\n" + "="*60)
    print(f"AGGREGATED BATCH RESULTS ({len(log_files)} Files Found)")
    print("="*60)
    print(f"Success Rate: {(success_count/len(log_files))*100:.1f}%")
    print(f"Avg Computation Time: {np.mean(comp_times):.4f}s (±{np.std(comp_times):.4f})")
    print(f"Avg Travel Time: {np.mean(trav_times):.2f}s")
    print(f"Avg Path Distance: {np.mean(distances):.2f}m")
    print(f"Avg Velocity: {np.mean(avg_speeds):.2f}m/s")
    print("="*60 + "\n")

    plot_batch_metrics(comp_times, trav_times, distances)
    plt.show()

def plot_rrt_results(log_data):
    """Plot the trajectory for a SINGLE run."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    meta = log_data['metadata']
    goal = meta['goal_pos']
    start = meta['start_pose']
    
    if 'planned_rrt_path' in log_data:
        px = [p['x'] for p in log_data['planned_rrt_path']]
        py = [p['y'] for p in log_data['planned_rrt_path']]
        ax.plot(px, py, 'b--', alpha=0.5, label='Planned Path')

    traj = log_data['executed_trajectory']
    tx, ty = [t['x'] for t in traj], [t['y'] for t in traj]
    ax.plot(tx, ty, 'r-', linewidth=2, label='Executed')
    ax.plot(start[0], start[1], 'bo', label='Start')
    ax.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')
    
    ax.set_title(f"Single Run: {meta.get('computation_time_sec', 0):.4f}s Solve")
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    return fig

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            log_data = json.load(f)
        plot_rrt_results(log_data)
        plt.show()
    elif ANALYZE_ALL:
        analyze_all_logs()
    else:
        log_data = load_latest_log()
        if log_data:
            plot_rrt_results(log_data)
            plt.show()

if __name__ == "__main__":
    main()