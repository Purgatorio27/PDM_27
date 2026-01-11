#!/usr/bin/env python3
"""
Analyzes the latest log files for RRT, MPC, and MPC+RRT comparison.
Generates a markdown table summary.
(No pandas dependency version)
"""

import json
import os
import glob
import numpy as np
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

def get_latest_log(pattern):
    search_pattern = os.path.join(LOG_DIR, pattern)
    files = glob.glob(search_pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def calculate_metrics(log_data):
    """Calculate extended metrics from log data."""
    metrics = {}
    
    # Defaults
    metrics['success'] = False
    metrics['time'] = 0.0
    metrics['path_length'] = 0.0
    metrics['avg_velocity'] = 0.0
    metrics['final_dist'] = 0.0
    metrics['min_clearance'] = 0.0
    metrics['collisions'] = 0
    metrics['solve_time_avg'] = 0.0

    if 'summary' in log_data:
        s = log_data['summary']
        metrics['success'] = s.get('goal_reached', s.get('final_distance_to_goal', 100) < 1.0)
        metrics['time'] = s.get('total_time_s', s.get('planning_time', 0))
        metrics['path_length'] = s.get('path_length', 0)
        metrics['avg_velocity'] = s.get('avg_speed', 0)
        metrics['final_dist'] = s.get('final_distance_to_goal', 0)
        metrics['min_clearance'] = s.get('min_obstacle_clearance', 0)
        metrics['collisions'] = s.get('collision_count', 0)
        metrics['solve_time_avg'] = s.get('avg_solve_time_ms', 0)
    
    # If full trajectory exists (MPC / MPC+RRT)
    if 'trajectory' in log_data:
        traj = log_data['trajectory']
        # Recalculate if summary missing
        if metrics['avg_velocity'] == 0 and len(traj) > 0:
            vs = []
            for t in traj:
                if 'v' in t: vs.append(t['v'])
                elif 'state' in t and 'v' in t['state']: vs.append(t['state']['v'])
            if vs:
                metrics['avg_velocity'] = sum(vs) / len(vs)
            
    return metrics

def print_table(headers, data):
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in data:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
            
    # Print headers
    header_row = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data
    for row in data:
        print(" | ".join(f"{str(v):<{w}}" for v, w in zip(row, widths)))

def main():
    print(f"Analyzing Logs in {LOG_DIR}")
    
    configs = [
        ("Pure RRT*", "rrt_comparison_*.json"),
        ("Pure MPC", "mpc_comparison_*.json"),
        ("MPC + RRT*", "mpc_rrt_comparison_*.json")
    ]
    
    headers = ['Method', 'Success', 'Time (s)', 'Path Len (m)', 'Avg Vel (m/s)', 'Min Clear (m)', 'Collisions']
    table_data = []
    
    for name, pattern in configs:
        log_file = get_latest_log(pattern)
        if not log_file:
            continue
            
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            m = calculate_metrics(data)
            
            row = [
                name,
                "YES" if m['success'] else "NO",
                f"{m['time']:.2f}",
                f"{m['path_length']:.2f}",
                f"{m['avg_velocity']:.2f}",
                f"{m['min_clearance']:.2f}",
                m['collisions']
            ]
            table_data.append(row)
        except Exception as e:
            print(f"Error parsing {pattern}: {e}")

    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print_table(headers, table_data)
    print("="*80)

if __name__ == "__main__":
    main()
