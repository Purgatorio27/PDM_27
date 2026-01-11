"""
Analyze and compare MPC, RRT*, and MPC+RRT* simulation logs.
Generates comparison plots and statistics.
"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from shared_config import MAP_SIZE, START_POS, GOAL_POS, STATIC_OBSTACLES
except ImportError:
    # Fallback if running from a different directory structure
    MAP_SIZE = 25.0
    START_POS = [2.0, 2.0]
    GOAL_POS = [23.0, 23.0]
    STATIC_OBSTACLES = []


def load_latest_logs():
    """Load the most recent MPC, RRT*, and MPC+RRT* logs."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

    mpc_logs = sorted(glob.glob(os.path.join(log_dir, 'mpc_comparison_*.json')))
    rrt_logs = sorted(glob.glob(os.path.join(log_dir, 'rrt_comparison_*.json')))
    mpc_rrt_logs = sorted(glob.glob(os.path.join(log_dir, 'mpc_rrt_comparison_*.json')))

    logs = {
        "MPC": None,
        "RRT*": None,
        "MPC+RRT*": None
    }

    if mpc_logs:
        with open(mpc_logs[-1], 'r') as f:
            logs["MPC"] = json.load(f)
        print(f"Loaded MPC log: {os.path.basename(mpc_logs[-1])}")

    if rrt_logs:
        with open(rrt_logs[-1], 'r') as f:
            logs["RRT*"] = json.load(f)
        print(f"Loaded RRT* log: {os.path.basename(rrt_logs[-1])}")

    if mpc_rrt_logs:
        with open(mpc_rrt_logs[-1], 'r') as f:
            logs["MPC+RRT*"] = json.load(f)
        print(f"Loaded MPC+RRT* log: {os.path.basename(mpc_rrt_logs[-1])}")

    return logs


def print_comparison_table(logs):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 90)
    print(f"{'METRIC':<30} {'MPC':>15} {'RRT*':>15} {'MPC+RRT*':>15}")
    print("=" * 90)

    def get_val(log, *keys, default="N/A"):
        if log is None: return default
        try:
            val = log
            for k in keys:
                val = val[k]
            return val
        except:
            return default

    # Define metrics
    # Format: (Label, (key_path...))
    # Special handling for time to support legacy logs
    
    rows = []
    
    # 1. Total Time
    times = []
    for name in ["MPC", "RRT*", "MPC+RRT*"]:
        log = logs[name]
        if log:
            # Try 'total_time_s' first (simulated time), then fall back
            t = get_val(log, 'summary', 'total_time_s', default=None)
            if t is None:
                # Fallback for old RRT logs
                t = get_val(log, 'planning_info', 'execution_time_s', default="N/A")
            times.append(t)
        else:
            times.append("N/A")
    rows.append(("Travel Time (s)", times))

    # 2. Path Length
    lengths = []
    for name in ["MPC", "RRT*", "MPC+RRT*"]:
        log = logs[name]
        l = get_val(log, 'summary', 'path_length', default=None)
        if l is None: # Fallback for RRT
            l = get_val(log, 'planning_info', 'path_length', default="N/A")
        lengths.append(l)
    rows.append(("Path Length (m)", lengths))

    # 3. Path Efficiency
    effs = []
    for name in ["MPC", "RRT*", "MPC+RRT*"]:
        log = logs[name]
        e = get_val(log, 'summary', 'path_efficiency', default=None)
        if e is None:
             e = get_val(log, 'planning_info', 'path_efficiency', default="N/A")
        if isinstance(e, (int, float)):
            e = e * 100
        effs.append(e)
    rows.append(("Path Efficiency (%)", effs))

    # 4. Success (Distance to Goal)
    dists = [get_val(logs[name], 'summary', 'final_distance_to_goal') for name in ["MPC", "RRT*", "MPC+RRT*"]]
    rows.append(("Final Dist to Goal (m)", dists))

    # 5. Min Clearance
    clears = [get_val(logs[name], 'summary', 'min_obstacle_clearance') for name in ["MPC", "RRT*", "MPC+RRT*"]]
    rows.append(("Min Obstacle Clear (m)", clears))

    # 6. Collisions
    cols = [get_val(logs[name], 'summary', 'collision_count') for name in ["MPC", "RRT*", "MPC+RRT*"]]
    rows.append(("Collisions", cols))

    # 7. Total Steps
    steps = [get_val(logs[name], 'summary', 'total_steps') for name in ["MPC", "RRT*", "MPC+RRT*"]]
    rows.append(("Total Steps", steps))
    
    # 8. Planning/Solve Time
    ptimes = []
    for name in ["MPC", "RRT*", "MPC+RRT*"]:
        log = logs[name]
        if name == "RRT*":
             t = get_val(log, 'planning_info', 'planning_time_ms', default="N/A")
        else:
             t = get_val(log, 'summary', 'total_solve_time_ms', default="N/A")
        ptimes.append(t)
    rows.append(("Comp. Time (ms)", ptimes))

    # Print rows
    for label, values in rows:
        row_str = f"{label:<30}"
        for v in values:
            if isinstance(v, (int, float)):
                row_str += f" {v:>14.2f} "
            else:
                row_str += f" {str(v):>15} "
        print(row_str)

    print("=" * 90)


def plot_trajectories(logs):
    """Plot trajectories on the same map."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw obstacles
    for obs in STATIC_OBSTACLES:
        circle = plt.Circle(obs['position'], obs['radius'], color='k', alpha=0.3)
        ax.add_patch(circle)

    colors = {"MPC": "blue", "RRT*": "green", "MPC+RRT*": "red"}
    styles = {"MPC": "-", "RRT*": "--", "MPC+RRT*": "-."}

    for name, log in logs.items():
        if log and log.get('trajectory'):
            xs = [p['x'] for p in log['trajectory']]
            ys = [p['y'] for p in log['trajectory']]
            ax.plot(xs, ys, color=colors[name], linestyle=styles[name], linewidth=2.5, label=name, alpha=0.8)
            # Mark end
            if xs:
                ax.plot(xs[-1], ys[-1], marker='x', color=colors[name], markersize=10)

    # Mark start and goal
    ax.plot(START_POS[0], START_POS[1], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(GOAL_POS[0], GOAL_POS[1], 'm*', markersize=15, label='Goal', zorder=5)

    ax.set_xlim(-1, MAP_SIZE + 1)
    ax.set_ylim(-1, MAP_SIZE + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_metrics_over_time(logs, metric_key, y_label, title):
    """Plot a specific metric from controls over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = {"MPC": "blue", "RRT*": "green", "MPC+RRT*": "red"}
    styles = {"MPC": "-", "RRT*": "--", "MPC+RRT*": "-."}

    has_data = False
    for name, log in logs.items():
        if log and log.get('controls'):
            steps = [c['step'] for c in log['controls']]
            # Access nested keys
            values = []
            for c in log['controls']:
                val = c
                try:
                    for k in metric_key:
                        val = val[k]
                    values.append(val)
                except:
                    values.append(0) # Fallback
            
            if values:
                ax.plot(steps, values, color=colors[name], linestyle=styles[name], linewidth=2, label=name, alpha=0.8)
                has_data = True

    if not has_data:
        plt.close(fig)
        return None

    if "Distance" in title:
         ax.axhline(y=0.5, color='gray', linestyle=':', label='Goal Tolerance')
    if "Clearance" in title:
         ax.axhline(y=0, color='r', linestyle='-', linewidth=1, label='Collision')

    ax.set_xlabel('Step')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
    
def plot_velocity_profiles(logs):
    """Plot velocity profiles."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = {"MPC": "blue", "RRT*": "green", "MPC+RRT*": "red"}
    styles = {"MPC": "-", "RRT*": "--", "MPC+RRT*": "-."}

    has_data = False
    for name, log in logs.items():
        if log and log.get('trajectory'):
            # Some logs store v in trajectory item, others in state dict
            steps = []
            vels = []
            for i, p in enumerate(log['trajectory']):
                steps.append(i)
                v = p.get('v')
                if v is None and 'state' in p:
                    v = p['state'].get('v')
                if v is None: v = 0
                vels.append(v)
            
            ax.plot(steps, vels, color=colors[name], linestyle=styles[name], linewidth=2, label=name, alpha=0.8)
            has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlabel('Step')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    print("Loading simulation logs...")
    logs = load_latest_logs()

    if all(l is None for l in logs.values()):
        print("ERROR: No logs found! Run simulations first.")
        return

    # Print comparison table
    print_comparison_table(logs)

    # Generate plots
    print("\nGenerating comparison plots...")
    figs = []
    
    # 1. Trajectories
    figs.append(("comparison_trajectory", plot_trajectories(logs)))
    
    # 2. Distance to Goal
    figs.append(("comparison_distance", plot_metrics_over_time(logs, ['distances', 'to_goal'], "Distance to Goal (m)", "Distance to Goal Over Time")))
    
    # 3. Obstacle Clearance
    figs.append(("comparison_clearance", plot_metrics_over_time(logs, ['distances', 'to_nearest_obstacle'], "Min Clearance (m)", "Obstacle Clearance Over Time")))
    
    # 4. Velocity
    figs.append(("comparison_velocity", plot_velocity_profiles(logs)))

    # Save plots
    output_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    for name, fig in figs:
        if fig:
            filename = f"{name}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")
            plt.close(fig) # free memory

    print("\nDone.")

if __name__ == "__main__":
    main()
