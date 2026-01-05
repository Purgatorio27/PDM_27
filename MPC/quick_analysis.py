#!/usr/bin/env python3
"""Quick analysis of the latest simulation log"""
import json
import numpy as np
import glob
import os

# Find latest log
log_dir = '/home/yanghongyi/PDM_27/logs'
log_files = glob.glob(os.path.join(log_dir, 'simulation_log_*.json'))
latest_file = max(log_files, key=os.path.getctime)
print(f"Analyzing: {latest_file}")

with open(latest_file, 'r') as f:
    data = json.load(f)

traj = data['trajectory']
controls = data['controls']
goal = data['metadata']['goal']

print('='*60)
print('SIMULATION ANALYSIS')
print('='*60)

# Basic stats
print(f'Total steps: {len(traj)}')
print(f'Total time: {len(traj) * 0.05:.2f}s')
print(f'Goal: {goal}')

# Final position
final = traj[-1]
dist_final = np.hypot(final['x'] - goal[0], final['y'] - goal[1])
print(f'Final position: ({final["x"]:.2f}, {final["y"]:.2f})')
print(f'Final distance to goal: {dist_final:.2f}m')

# Velocity analysis
velocities = [c['state']['v'] for c in controls]
print(f'\nVelocity: min={min(velocities):.2f}, max={max(velocities):.2f}, avg={np.mean(velocities):.2f}')

# Split into phases
quarters = len(velocities) // 4
for i in range(4):
    start = i * quarters
    end = (i + 1) * quarters
    avg_v = np.mean(velocities[start:end])
    print(f'  Quarter {i+1} avg velocity: {avg_v:.2f} m/s')

# Heading analysis
print('\n--- HEADING ANALYSIS AT KEY POINTS ---')
key_points = [0, len(controls)//4, len(controls)//2, 3*len(controls)//4, len(controls)-1]
for step_idx in key_points:
    c = controls[step_idx]
    x, y = c['state']['x'], c['state']['y']
    psi = c['state']['psi']
    goal_angle = np.arctan2(goal[1] - y, goal[0] - x)
    heading_error = goal_angle - psi
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    print(f'Step {step_idx}: pos=({x:.1f},{y:.1f}), heading={np.degrees(psi):.1f}°, goal_dir={np.degrees(goal_angle):.1f}°, error={np.degrees(heading_error):.1f}°')

# Heading error over time
heading_errors = []
for c in controls:
    x, y = c['state']['x'], c['state']['y']
    psi = c['state']['psi']
    goal_angle = np.arctan2(goal[1] - y, goal[0] - x)
    error = goal_angle - psi
    error = np.arctan2(np.sin(error), np.cos(error))
    heading_errors.append(np.degrees(error))

print(f'\nHeading error: min={min(heading_errors):.1f}°, max={max(heading_errors):.1f}°, avg={np.mean(np.abs(heading_errors)):.1f}°')

# Large heading errors (problem areas)
print('\n--- LARGE HEADING ERROR MOMENTS (>30°) ---')
large_error_count = 0
for i, err in enumerate(heading_errors):
    if abs(err) > 30:
        large_error_count += 1
        if large_error_count <= 5:
            c = controls[i]
            print(f'  Step {c["step"]}: pos=({c["state"]["x"]:.1f},{c["state"]["y"]:.1f}), error={err:.1f}°, steer={c["control"]["steering_deg"]:.1f}°')
print(f'Total steps with |heading_error| > 30°: {large_error_count}')

# Steering analysis
steerings = [c['control']['steering_deg'] for c in controls]
print(f'\nSteering: min={min(steerings):.1f}°, max={max(steerings):.1f}°, avg={np.mean(steerings):.1f}°')

# Cost analysis
costs = [c['solver']['cost'] for c in controls if c['solver']['cost'] > 0]
if costs:
    print(f'\nCost: min={min(costs):.2f}, max={max(costs):.2f}')
    for i in range(4):
        start = i * len(costs) // 4
        end = (i + 1) * len(costs) // 4
        print(f'  Quarter {i+1} avg cost: {np.mean(costs[start:end]):.2f}')

# Find slowest section
print('\n--- SLOWEST SECTIONS ---')
window = 50
min_avg_v = float('inf')
min_start = 0
for i in range(len(velocities) - window):
    avg_v = np.mean(velocities[i:i+window])
    if avg_v < min_avg_v:
        min_avg_v = avg_v
        min_start = i
c = controls[min_start]
print(f'Slowest 50-step window: steps {min_start}-{min_start+window}')
print(f'  Avg velocity: {min_avg_v:.2f} m/s')
print(f'  Position: ({c["state"]["x"]:.1f}, {c["state"]["y"]:.1f})')

# Solver stats
solver_failures = sum(1 for c in controls if c['solver']['status'] != 0)
print(f'\nSolver failures: {solver_failures}/{len(controls)} ({100*solver_failures/len(controls):.1f}%)')

# Trajectory shape analysis
print('\n--- TRAJECTORY SHAPE ANALYSIS ---')
x_coords = [t['x'] for t in traj]
y_coords = [t['y'] for t in traj]

# Calculate path length vs straight line distance
path_length = sum(np.hypot(x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i]) for i in range(len(x_coords)-1))
straight_dist = np.hypot(x_coords[-1] - x_coords[0], y_coords[-1] - y_coords[0])
print(f'Path length: {path_length:.2f}m')
print(f'Straight line distance: {straight_dist:.2f}m')
print(f'Path efficiency: {100 * straight_dist / path_length:.1f}%')

# Curvature analysis - detect big arcs
print('\n--- ARC DETECTION ---')
# Calculate cumulative heading change
total_heading_change = 0
for i in range(1, len(traj)):
    delta_psi = traj[i]['psi'] - traj[i-1]['psi']
    delta_psi = np.arctan2(np.sin(delta_psi), np.cos(delta_psi))
    total_heading_change += delta_psi

print(f'Total cumulative heading change: {np.degrees(total_heading_change):.1f}°')
print(f'(Positive = counter-clockwise, Negative = clockwise)')
