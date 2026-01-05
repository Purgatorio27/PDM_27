"""
Simulation Log Analyzer and Visualizer
Run this script to analyze and visualize the simulation logs.

Usage:
    python analyze_log.py                    # Analyze the latest log
    python analyze_log.py logs/filename.json # Analyze a specific log
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import glob


def load_latest_log():
    """Load the most recent simulation log"""
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    log_files = glob.glob(os.path.join(log_dir, 'simulation_log_*.json'))
    
    if not log_files:
        print("No log files found in logs/ directory")
        return None
    
    latest_file = max(log_files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def load_log(filepath):
    """Load a specific log file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_trajectory(log_data, show_mpc_predictions=True, prediction_interval=50):
    """Plot the vehicle trajectory with obstacles"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Extract trajectory data
    traj = log_data['trajectory']
    x = [t['x'] for t in traj]
    y = [t['y'] for t in traj]
    v = [t['v'] for t in traj]
    
    # Plot static obstacles
    for obs in log_data['metadata']['static_obstacles']:
        circle = Circle(obs['position'], obs['radius'], 
                       color='gray', alpha=0.7, label='Static Obstacle')
        ax.add_patch(circle)
    
    # Plot dynamic obstacle initial positions
    for i, obs in enumerate(log_data['metadata']['dynamic_obstacles_initial']):
        circle = Circle(obs['start_pos'], obs['radius'], 
                       color='yellow', alpha=0.5, label=f'Dyn Obs {i+1} Start')
        ax.add_patch(circle)
    
    # Plot goal
    goal = log_data['metadata']['goal']
    ax.plot(goal[0], goal[1], 'g*', markersize=20, label='Goal')
    
    # Plot trajectory colored by velocity
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(min(v), max(v))
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linewidth=3)
    lc.set_array(np.array(v[:-1]))
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax, label='Velocity (m/s)')
    
    # Plot start point
    ax.plot(x[0], y[0], 'bo', markersize=10, label='Start')
    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    
    # Plot MPC predictions at intervals
    if show_mpc_predictions and 'mpc_predictions' in log_data:
        for pred in log_data['mpc_predictions'][::prediction_interval]:
            if pred['predicted_trajectory']:
                pred_x = [p[0] for p in pred['predicted_trajectory']]
                pred_y = [p[1] for p in pred['predicted_trajectory']]
                ax.plot(pred_x, pred_y, 'c--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Vehicle Trajectory')
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    margin = 3
    ax.set_xlim(min(x) - margin, max(max(x), goal[0]) + margin)
    ax.set_ylim(min(y) - margin, max(max(y), goal[1]) + margin)
    
    plt.tight_layout()
    return fig


def plot_controls_and_state(log_data):
    """Plot control inputs and vehicle state over time"""
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    
    controls = log_data['controls']
    steps = [c['step'] for c in controls]
    times = [c['time'] for c in controls]
    
    # Velocity
    v = [c['state']['v'] for c in controls]
    axes[0, 0].plot(times, v, 'b-', linewidth=1.5)
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].set_title('Vehicle Velocity')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=log_data['metadata']['max_speed'] * 0.75, color='g', linestyle='--', label='v_ref (far)')
    axes[0, 0].axhline(y=log_data['metadata']['max_speed'] * 0.3, color='r', linestyle='--', label='v_ref (near)')
    axes[0, 0].legend()
    
    # Heading
    psi = [c['state']['psi_deg'] for c in controls]
    axes[0, 1].plot(times, psi, 'b-', linewidth=1.5)
    axes[0, 1].set_ylabel('Heading (deg)')
    axes[0, 1].set_title('Vehicle Heading')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Steering angle
    steer = [c['control']['steering_deg'] for c in controls]
    axes[1, 0].plot(times, steer, 'r-', linewidth=1.5)
    axes[1, 0].set_ylabel('Steering (deg)')
    axes[1, 0].set_title('Steering Angle')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Acceleration
    accel = [c['control']['acceleration'] for c in controls]
    axes[1, 1].plot(times, accel, 'g-', linewidth=1.5)
    axes[1, 1].set_ylabel('Acceleration (m/s²)')
    axes[1, 1].set_title('Acceleration')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Distance to goal
    dist_goal = [c['distances'].get('to_goal', 0) for c in controls]
    axes[2, 0].plot(times, dist_goal, 'm-', linewidth=1.5)
    axes[2, 0].set_ylabel('Distance (m)')
    axes[2, 0].set_title('Distance to Goal')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Distance to nearest obstacle
    dist_obs = [c['distances'].get('to_nearest_obstacle', 0) for c in controls]
    axes[2, 1].plot(times, dist_obs, 'orange', linewidth=1.5)
    axes[2, 1].axhline(y=0, color='r', linestyle='--', label='Collision boundary')
    axes[2, 1].set_ylabel('Distance (m)')
    axes[2, 1].set_title('Distance to Nearest Obstacle')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    # Solver cost
    costs = [c['solver']['cost'] for c in controls]
    axes[3, 0].plot(times, costs, 'purple', linewidth=1.5)
    axes[3, 0].set_ylabel('Cost')
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_title('MPC Solver Cost')
    axes[3, 0].grid(True, alpha=0.3)
    
    # Solver status
    status = [c['solver']['status'] for c in controls]
    axes[3, 1].plot(times, status, 'k-', linewidth=1.5, drawstyle='steps-post')
    axes[3, 1].set_ylabel('Status (0=OK)')
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].set_title('Solver Status')
    axes[3, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_heading_analysis(log_data):
    """Analyze heading behavior - key for debugging turning issues"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    controls = log_data['controls']
    times = [c['time'] for c in controls]
    goal = log_data['metadata']['goal']
    
    # Current heading
    psi = [c['state']['psi'] for c in controls]
    
    # Compute ideal heading to goal at each step
    x = [c['state']['x'] for c in controls]
    y = [c['state']['y'] for c in controls]
    goal_angles = [np.arctan2(goal[1] - yi, goal[0] - xi) for xi, yi in zip(x, y)]
    
    # Heading error
    heading_errors = []
    for p, g in zip(psi, goal_angles):
        error = g - p
        error = np.arctan2(np.sin(error), np.cos(error))  # normalize
        heading_errors.append(np.degrees(error))
    
    # Plot 1: Heading vs Goal direction
    axes[0, 0].plot(times, np.degrees(psi), 'b-', label='Vehicle Heading', linewidth=1.5)
    axes[0, 0].plot(times, np.degrees(goal_angles), 'g--', label='Goal Direction', linewidth=1.5)
    axes[0, 0].set_ylabel('Angle (deg)')
    axes[0, 0].set_title('Heading vs Goal Direction')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Heading error
    axes[0, 1].plot(times, heading_errors, 'r-', linewidth=1.5)
    axes[0, 1].axhline(y=0, color='k', linestyle='--')
    axes[0, 1].set_ylabel('Heading Error (deg)')
    axes[0, 1].set_title('Heading Error (Goal - Vehicle)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Steering angle vs heading error
    steer = [c['control']['steering_deg'] for c in controls]
    axes[1, 0].scatter(heading_errors, steer, c=times, cmap='viridis', alpha=0.6, s=10)
    axes[1, 0].set_xlabel('Heading Error (deg)')
    axes[1, 0].set_ylabel('Steering Angle (deg)')
    axes[1, 0].set_title('Steering Response to Heading Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: XY trajectory colored by heading error
    ax = axes[1, 1]
    scatter = ax.scatter(x, y, c=np.abs(heading_errors), cmap='RdYlGn_r', s=20)
    plt.colorbar(scatter, ax=ax, label='|Heading Error| (deg)')
    
    # Add obstacles
    for obs in log_data['metadata']['static_obstacles']:
        circle = Circle(obs['position'], obs['radius'], color='gray', alpha=0.5)
        ax.add_patch(circle)
    
    ax.plot(goal[0], goal[1], 'g*', markersize=15)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Colored by Heading Error')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_summary(log_data):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SIMULATION LOG SUMMARY")
    print("="*60)
    
    meta = log_data['metadata']
    print(f"\nGoal: {meta['goal']}")
    print(f"Max Speed: {meta['max_speed']} m/s")
    print(f"Time Step: {meta['dt']} s")
    
    controls = log_data['controls']
    traj = log_data['trajectory']
    
    print(f"\nTotal Steps: {len(traj)}")
    print(f"Total Time: {len(traj) * meta['dt']:.2f} s")
    
    # Final position
    if traj:
        final = traj[-1]
        dist_to_goal = np.hypot(final['x'] - meta['goal'][0], final['y'] - meta['goal'][1])
        print(f"\nFinal Position: ({final['x']:.2f}, {final['y']:.2f})")
        print(f"Final Distance to Goal: {dist_to_goal:.2f} m")
    
    # Velocity stats
    velocities = [c['state']['v'] for c in controls]
    print(f"\nVelocity Statistics:")
    print(f"  Min: {min(velocities):.2f} m/s")
    print(f"  Max: {max(velocities):.2f} m/s")
    print(f"  Mean: {np.mean(velocities):.2f} m/s")
    
    # Solver stats
    solver_failures = sum(1 for c in controls if c['solver']['status'] != 0)
    print(f"\nSolver Statistics:")
    print(f"  Failures: {solver_failures}/{len(controls)} ({100*solver_failures/len(controls):.1f}%)")
    
    solve_times = [c['solver']['solve_time_ms'] for c in controls]
    print(f"  Solve Time: min={min(solve_times):.2f}ms, max={max(solve_times):.2f}ms, avg={np.mean(solve_times):.2f}ms")
    
    # Minimum obstacle distance
    min_obs_dists = [c['distances'].get('to_nearest_obstacle', float('inf')) for c in controls]
    min_obs_dist = min(min_obs_dists)
    print(f"\nMinimum Obstacle Clearance: {min_obs_dist:.2f} m")
    if min_obs_dist < 0:
        print("  WARNING: Collision detected!")


def main():
    # Load log file
    if len(sys.argv) > 1:
        log_data = load_log(sys.argv[1])
    else:
        log_data = load_latest_log()
    
    if log_data is None:
        return
    
    # Print summary
    print_summary(log_data)
    
    # Create plots
    print("\nGenerating plots...")
    
    fig1 = plot_trajectory(log_data, show_mpc_predictions=True, prediction_interval=20)
    fig2 = plot_controls_and_state(log_data)
    fig3 = plot_heading_analysis(log_data)
    
    plt.show()


if __name__ == "__main__":
    main()
