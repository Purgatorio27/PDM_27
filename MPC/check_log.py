#!/usr/bin/env python3
"""Quick check of latest log"""
import json
import glob
import os

log_dir = '/home/PDMproject/PDM_27/logs'
log_files = glob.glob(os.path.join(log_dir, 'simulation_log_*.json'))
latest_file = max(log_files, key=os.path.getctime)
print(f"Analyzing: {latest_file}")

with open(latest_file) as f:
    data = json.load(f)

controls = data['controls']
print(f'Steps: {len(controls)}')

costs = [c['solver']['cost'] for c in controls]
print(f'Cost range: {min(costs):.4f} - {max(costs):.4f}')

vels = [c['state']['v'] for c in controls]
print(f'Velocity range: {min(vels):.2f} - {max(vels):.2f}')
print(f'Avg velocity: {sum(vels)/len(vels):.2f}')

# Check trajectory shape
traj = data['trajectory']
x0, y0 = traj[0]['x'], traj[0]['y']
xf, yf = traj[-1]['x'], traj[-1]['y']
path_len = sum(((traj[i+1]['x']-traj[i]['x'])**2 + (traj[i+1]['y']-traj[i]['y'])**2)**0.5 for i in range(len(traj)-1))
straight_dist = ((xf-x0)**2 + (yf-y0)**2)**0.5
print(f'Path length: {path_len:.1f}m, Straight: {straight_dist:.1f}m, Efficiency: {100*straight_dist/path_len:.1f}%')
