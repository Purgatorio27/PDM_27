#!/usr/bin/env python3
"""
Debug script: Compare RRT-driven vs MPC-driven vehicle control.

Strategy:
1. Plan RRT path from start to goal
2. Drive vehicle using Pure Pursuit following RRT path
3. At each step, also compute what MPC would output
4. Compare the two control signals and analyze differences
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../MPC'))

import numpy as np
import math
import time
import json
from datetime import datetime

# Import shared config
from shared_config import (
    MAP_SIZE, START_POS, GOAL_POS, START_YAW,
    STATIC_OBSTACLES, VEHICLE_RADIUS, VEHICLE_LENGTH, VEHICLE_WIDTH,
    MAX_SPEED, MIN_SPEED, GOAL_TOLERANCE
)

# MPC imports
from MPC_RRT import MPC_RRT_Controller, Waypoint

# Vehicle parameters
LF = VEHICLE_LENGTH / 2
LR = VEHICLE_LENGTH / 2
CONTROLLER_DT = 0.05


class PurePursuitController:
    """Simple Pure Pursuit controller for path following."""
    
    def __init__(self, waypoints, lookahead_distance=3.0, target_speed=2.0):
        self.waypoints = waypoints
        self.lookahead_distance = lookahead_distance
        self.target_speed = target_speed
        self.current_waypoint_idx = 0
        self.L = VEHICLE_LENGTH  # Wheelbase
        
    def find_lookahead_point(self, x, y, psi):
        """Find the lookahead point on the path."""
        # Start from current waypoint
        for i in range(self.current_waypoint_idx, len(self.waypoints)):
            wp = self.waypoints[i]
            dist = math.hypot(wp.x - x, wp.y - y)
            
            if dist >= self.lookahead_distance:
                return wp, i
            
            # Update current waypoint if we've passed it
            if dist < 1.5:
                self.current_waypoint_idx = min(i + 1, len(self.waypoints) - 1)
        
        # Return last waypoint if no lookahead found
        return self.waypoints[-1], len(self.waypoints) - 1
    
    def compute_control(self, state):
        """Compute steering and acceleration using Pure Pursuit."""
        x, y, v, psi = state
        
        # Find lookahead point
        target_wp, wp_idx = self.find_lookahead_point(x, y, psi)
        
        # Compute steering angle using Pure Pursuit formula
        dx = target_wp.x - x
        dy = target_wp.y - y
        
        # Transform to vehicle frame
        alpha = math.atan2(dy, dx) - psi
        
        # Normalize alpha to [-pi, pi]
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi
        
        # Pure Pursuit steering: delta = atan(2 * L * sin(alpha) / Ld)
        Ld = math.hypot(dx, dy)
        if Ld < 0.1:
            Ld = 0.1
        
        steering = math.atan2(2.0 * self.L * math.sin(alpha), Ld)
        
        # Limit steering angle
        max_steer = 0.5  # ~30 degrees
        steering = np.clip(steering, -max_steer, max_steer)
        
        # Simple speed control
        if v < self.target_speed:
            accel = 1.0
        else:
            accel = -0.5
        
        # Slow down near obstacles
        dist_to_goal = math.hypot(GOAL_POS[0] - x, GOAL_POS[1] - y)
        if dist_to_goal < 3.0:
            accel = -1.0 if v > 0.5 else 0.0
        
        return np.array([steering, accel]), {
            'target_wp': (target_wp.x, target_wp.y),
            'wp_idx': wp_idx,
            'lookahead_dist': Ld,
            'alpha': alpha
        }


def vehicle_simulate_engine(state, control, dt):
    """Kinematic bicycle model update."""
    x, y, v, psi = state
    delta, a = control

    x_new = x + v * math.cos(psi) * dt
    y_new = y + v * math.sin(psi) * dt

    v_new = v + a * dt
    v_new = np.clip(v_new, MIN_SPEED, MAX_SPEED)

    beta = math.atan((LR / (LF + LR)) * math.tan(delta))
    psi_new = psi + (v / LR) * math.sin(beta) * dt

    return np.array([x_new, y_new, v_new, psi_new])


def compute_min_obstacle_distance(car_pos, obstacles):
    """Compute minimum distance to any obstacle."""
    min_dist = float('inf')
    closest_obs_idx = -1
    for i, obs in enumerate(obstacles):
        ox, oy = obs['position']
        dist = math.hypot(car_pos[0] - ox, car_pos[1] - oy) - obs['radius'] - VEHICLE_RADIUS
        if dist < min_dist:
            min_dist = dist
            closest_obs_idx = i
    return min_dist, closest_obs_idx


def run_comparison():
    """Run comparison between RRT-driven and MPC-driven control."""
    
    print("=" * 70)
    print("RRT vs MPC Control Comparison")
    print("=" * 70)
    
    # Initialize state
    initial_state = np.array([START_POS[0], START_POS[1], 0.5, START_YAW])
    
    # Create MPC+RRT controller
    static_obstacles = list(STATIC_OBSTACLES)
    
    print("\nInitializing MPC+RRT controller...")
    mpc_rrt_controller = MPC_RRT_Controller(
        start_state=list(initial_state),
        goal=GOAL_POS,
        static_obstacles=static_obstacles,
        horizon=20,
        dt=0.2
    )
    
    if not mpc_rrt_controller.planning_success:
        print("RRT planning failed!")
        return
    
    print(f"RRT path: {len(mpc_rrt_controller.waypoints)} waypoints")
    
    # Create Pure Pursuit controller using RRT waypoints
    pp_controller = PurePursuitController(
        waypoints=mpc_rrt_controller.waypoints,
        lookahead_distance=3.0,
        target_speed=2.0
    )
    
    # Simulation logs
    rrt_log = []
    mpc_log = []
    comparison_log = []
    
    # Run two simulations: one with RRT (Pure Pursuit), one with MPC
    
    print("\n" + "=" * 70)
    print("Phase 1: RRT-driven simulation (Pure Pursuit following RRT path)")
    print("=" * 70)
    
    # RRT-driven simulation
    car_state_rrt = initial_state.copy()
    max_steps = 800
    
    for step in range(max_steps):
        # Compute Pure Pursuit control
        pp_control, pp_info = pp_controller.compute_control(car_state_rrt)
        
        # Also compute what MPC would output (without using it)
        try:
            mpc_control, mpc_status, mpc_debug = mpc_rrt_controller.solve(
                list(car_state_rrt), 
                debug=False
            )
        except Exception as e:
            mpc_control = np.array([0.0, 0.0])
            mpc_status = -1
            mpc_debug = {'error': str(e)}
        
        # Compute obstacle distance
        min_dist, closest_obs = compute_min_obstacle_distance(car_state_rrt[:2], static_obstacles)
        dist_to_goal = math.hypot(car_state_rrt[0] - GOAL_POS[0], car_state_rrt[1] - GOAL_POS[1])
        
        # Log comparison data
        comparison_log.append({
            'step': step,
            'state': {
                'x': float(car_state_rrt[0]),
                'y': float(car_state_rrt[1]),
                'v': float(car_state_rrt[2]),
                'psi': float(car_state_rrt[3])
            },
            'rrt_control': {
                'steering': float(pp_control[0]),
                'steering_deg': float(np.degrees(pp_control[0])),
                'accel': float(pp_control[1])
            },
            'mpc_control': {
                'steering': float(mpc_control[0]),
                'steering_deg': float(np.degrees(mpc_control[0])),
                'accel': float(mpc_control[1])
            },
            'control_diff': {
                'steering_diff': float(mpc_control[0] - pp_control[0]),
                'steering_diff_deg': float(np.degrees(mpc_control[0] - pp_control[0])),
                'accel_diff': float(mpc_control[1] - pp_control[1])
            },
            'distances': {
                'to_goal': float(dist_to_goal),
                'to_obstacle': float(min_dist),
                'closest_obs': int(closest_obs)
            },
            'mpc_debug': {
                'solver_status': mpc_status,
                'current_wp': mpc_debug.get('current_waypoint_idx', -1),
                'deviation': mpc_debug.get('deviation', 0.0)
            },
            'pp_info': {
                'target_wp': pp_info['target_wp'],
                'wp_idx': pp_info['wp_idx'],
                'lookahead_dist': pp_info['lookahead_dist']
            }
        })
        
        # Print significant differences
        steer_diff = abs(mpc_control[0] - pp_control[0])
        if steer_diff > 0.1 or min_dist < 2.5:  # Significant steering difference or near obstacle
            print(f"Step {step}: pos=({car_state_rrt[0]:.2f}, {car_state_rrt[1]:.2f}) "
                  f"v={car_state_rrt[2]:.2f} dist_obs={min_dist:.2f}m "
                  f"RRT_steer={np.degrees(pp_control[0]):.1f}° "
                  f"MPC_steer={np.degrees(mpc_control[0]):.1f}° "
                  f"diff={np.degrees(steer_diff):.1f}°")
        
        # Check collision (using RRT path)
        if min_dist < 0:
            print(f"  *** RRT COLLISION at step {step}! clearance={min_dist:.3f}m obs={closest_obs}")
        
        # Check goal
        if dist_to_goal < GOAL_TOLERANCE:
            print(f"\n*** Goal reached at step {step}! ***")
            break
        
        # Apply RRT control (Pure Pursuit)
        car_state_rrt = vehicle_simulate_engine(car_state_rrt, pp_control, CONTROLLER_DT)
    
    rrt_final_dist = math.hypot(car_state_rrt[0] - GOAL_POS[0], car_state_rrt[1] - GOAL_POS[1])
    rrt_collisions = sum(1 for c in comparison_log if c['distances']['to_obstacle'] < 0)
    
    print(f"\nRRT-driven results:")
    print(f"  Final distance to goal: {rrt_final_dist:.2f}m")
    print(f"  Collisions: {rrt_collisions}")
    
    # Analyze control differences
    print("\n" + "=" * 70)
    print("Control Signal Analysis")
    print("=" * 70)
    
    # Find steps with largest steering differences
    steer_diffs = [(c['step'], c['control_diff']['steering_diff_deg'], c['state'], c['distances']) 
                   for c in comparison_log]
    steer_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop 10 largest steering differences (MPC - RRT):")
    for step, diff, state, dists in steer_diffs[:10]:
        print(f"  Step {step}: pos=({state['x']:.2f}, {state['y']:.2f}) "
              f"v={state['v']:.2f} obs_dist={dists['to_obstacle']:.2f}m "
              f"steer_diff={diff:+.1f}°")
    
    # Analyze near-obstacle behavior
    print("\nNear-obstacle analysis (dist < 3m):")
    near_obs_steps = [c for c in comparison_log if c['distances']['to_obstacle'] < 3.0]
    if near_obs_steps:
        avg_mpc_steer = np.mean([c['mpc_control']['steering_deg'] for c in near_obs_steps])
        avg_rrt_steer = np.mean([c['rrt_control']['steering_deg'] for c in near_obs_steps])
        avg_steer_diff = np.mean([c['control_diff']['steering_diff_deg'] for c in near_obs_steps])
        
        print(f"  Steps near obstacle: {len(near_obs_steps)}")
        print(f"  Avg MPC steering: {avg_mpc_steer:.1f}°")
        print(f"  Avg RRT steering: {avg_rrt_steer:.1f}°")
        print(f"  Avg difference: {avg_steer_diff:+.1f}°")
        
        # Check which obstacle
        obs_counts = {}
        for c in near_obs_steps:
            obs_idx = c['distances']['closest_obs']
            obs_counts[obs_idx] = obs_counts.get(obs_idx, 0) + 1
        print(f"  Closest obstacles: {obs_counts}")
    
    # Save detailed log
    log_file = f"/home/yanghongyi/PDM_27/logs/rrt_vs_mpc_debug_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(log_file, 'w') as f:
        json.dump({
            'config': {
                'rrt_lookahead': pp_controller.lookahead_distance,
                'rrt_target_speed': pp_controller.target_speed,
                'mpc_horizon': 20,
                'mpc_dt': 0.2
            },
            'results': {
                'rrt_final_dist': rrt_final_dist,
                'rrt_collisions': rrt_collisions
            },
            'comparison': comparison_log
        }, f, indent=2)
    
    print(f"\nDetailed log saved to: {log_file}")
    
    # Now run MPC-driven simulation for comparison
    print("\n" + "=" * 70)
    print("Phase 2: MPC-driven simulation")
    print("=" * 70)
    
    # Re-initialize controller
    mpc_rrt_controller2 = MPC_RRT_Controller(
        start_state=list(initial_state),
        goal=GOAL_POS,
        static_obstacles=static_obstacles,
        horizon=20,
        dt=0.2
    )
    
    car_state_mpc = initial_state.copy()
    mpc_only_log = []
    
    for step in range(max_steps):
        try:
            mpc_control, mpc_status, mpc_debug = mpc_rrt_controller2.solve(
                list(car_state_mpc), 
                debug=False
            )
        except Exception as e:
            mpc_control = np.array([0.0, 0.0])
        
        min_dist, closest_obs = compute_min_obstacle_distance(car_state_mpc[:2], static_obstacles)
        dist_to_goal = math.hypot(car_state_mpc[0] - GOAL_POS[0], car_state_mpc[1] - GOAL_POS[1])
        
        mpc_only_log.append({
            'step': step,
            'x': float(car_state_mpc[0]),
            'y': float(car_state_mpc[1]),
            'v': float(car_state_mpc[2]),
            'min_dist': float(min_dist),
            'collision': min_dist < 0
        })
        
        if min_dist < 0:
            print(f"  *** MPC COLLISION at step {step}! pos=({car_state_mpc[0]:.2f}, {car_state_mpc[1]:.2f}) "
                  f"clearance={min_dist:.3f}m obs={closest_obs}")
        
        if dist_to_goal < GOAL_TOLERANCE:
            print(f"\n*** Goal reached at step {step}! ***")
            break
        
        car_state_mpc = vehicle_simulate_engine(car_state_mpc, mpc_control, CONTROLLER_DT)
    
    mpc_final_dist = math.hypot(car_state_mpc[0] - GOAL_POS[0], car_state_mpc[1] - GOAL_POS[1])
    mpc_collisions = sum(1 for c in mpc_only_log if c['collision'])
    
    print(f"\nMPC-driven results:")
    print(f"  Final distance to goal: {mpc_final_dist:.2f}m")
    print(f"  Collisions: {mpc_collisions}")
    
    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"                    RRT-driven    MPC-driven")
    print(f"  Final dist:       {rrt_final_dist:.2f}m        {mpc_final_dist:.2f}m")
    print(f"  Collisions:       {rrt_collisions}             {mpc_collisions}")
    print("=" * 70)


if __name__ == '__main__':
    run_comparison()
