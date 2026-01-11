"""
MPC+RRT simulation for comparison.
Uses RRT* for global path planning and MPC for local trajectory tracking.
Uses shared_config.py for identical environment setup.

Usage:
    python mpc_rrt_comparison.py          # GUI mode
    python mpc_rrt_comparison.py --headless  # Headless mode (for testing)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MPC'))

# Check for headless mode
HEADLESS = '--headless' in sys.argv

import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import json
from datetime import datetime

from shared_config import (
    MAP_SIZE, START_POS, GOAL_POS, START_YAW,
    STATIC_OBSTACLES, VEHICLE_RADIUS, GOAL_TOLERANCE,
    MAX_SPEED, MIN_SPEED, VEHICLE_LENGTH, VEHICLE_WIDTH
)

# Patch MPC Config before importing MPC_RRT
import MPC.Config as mpc_cfg
mpc_cfg.goal = GOAL_POS
mpc_cfg.vehicle_length = VEHICLE_LENGTH
mpc_cfg.vehicle_width = VEHICLE_WIDTH
mpc_cfg.vehicle_radius = VEHICLE_RADIUS
mpc_cfg.lf = VEHICLE_LENGTH * 0.5
mpc_cfg.lr = VEHICLE_LENGTH * 0.5

# Import MPC_RRT controller (after config patch)
from MPC.MPC_RRT import MPC_RRT_Controller, create_mpc_rrt_controller


# ============== MPC+RRT Parameters ==============
CONTROLLER_DT = 0.05  # 20Hz control
MAX_STEPS = 1200  # Max simulation steps (Increased for 40x40 map)
MPC_HORIZON = 20  # Balanced for path following and planning
MPC_DT = 0.2

# Vehicle parameters
LF = VEHICLE_LENGTH * 0.5
LR = VEHICLE_LENGTH * 0.5
MAX_ACCELERATION = 3.0
MAX_DECELERATION = -3.0
MAX_STEERING_ANGLE = np.radians(45)


def setup_environment(headless=False):
    """Initialize PyBullet with 25x25 environment."""
    if headless:
        physicsClient = p.connect(p.DIRECT)
    else:
        physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(CONTROLLER_DT)

    # Load ground plane
    p.loadURDF("plane.urdf", [0, 0, 0])

    # Camera setup for 25x25 map (GUI only)
    if not headless:
        p.resetDebugVisualizerCamera(
            cameraDistance=20,
            cameraYaw=0,
            cameraPitch=-89.9,
            cameraTargetPosition=[MAP_SIZE/2, MAP_SIZE/2, 0]
        )

    return physicsClient


def create_vehicle():
    """Create vehicle in PyBullet."""
    car_half_extents = [VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2, 0.05]
    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=VEHICLE_RADIUS)
    car_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=car_half_extents, rgbaColor=[1, 0, 0, 1])

    car_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=car_visual,
        basePosition=[START_POS[0], START_POS[1], 0.1]
    )
    return car_id


def create_obstacles():
    """Create static obstacles from shared config."""
    obstacle_ids = []

    # Boundary walls
    wall_height = 1.0
    wall_thickness = 0.2
    wall_color = [0.6, 0.6, 0.6, 1]

    walls = [
        {'pos': [MAP_SIZE/2, 0], 'dim': [MAP_SIZE/2, wall_thickness]},
        {'pos': [MAP_SIZE/2, MAP_SIZE], 'dim': [MAP_SIZE/2, wall_thickness]},
        {'pos': [0, MAP_SIZE/2], 'dim': [wall_thickness, MAP_SIZE/2]},
        {'pos': [MAP_SIZE, MAP_SIZE/2], 'dim': [wall_thickness, MAP_SIZE/2]},
    ]

    for wall in walls:
        half_ext = [wall['dim'][0], wall['dim'][1], wall_height/2]
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=wall_color),
            basePosition=[wall['pos'][0], wall['pos'][1], wall_height/2]
        )

    # Circle obstacles
    for obs in STATIC_OBSTACLES:
        pos = obs['position']
        radius = obs['radius']

        col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=1.0)
        vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=1.0, rgbaColor=[0, 0, 1, 0.8])

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[pos[0], pos[1], 0.5]
        )
        obstacle_ids.append(body_id)

    # Goal marker
    goal_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 0, 0.8])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_shape, basePosition=[GOAL_POS[0], GOAL_POS[1], 0.1])

    return obstacle_ids


def get_mpc_obstacles():
    """Convert shared obstacles to MPC format."""
    obstacles = []
    for obs in STATIC_OBSTACLES:
        obstacles.append({
            'position': obs['position'],
            'type': 'circle',
            'radius': obs['radius']
        })
    return obstacles


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


def visualize_rrt_path(controller, path_lines_ids=None):
    """Visualize RRT path and waypoints in PyBullet."""
    # Remove old lines
    if path_lines_ids:
        for line_id in path_lines_ids:
            try:
                p.removeUserDebugItem(line_id)
            except:
                pass
    
    new_line_ids = []
    vis_data = controller.get_path_visualization()
    
    # Draw RRT path (green lines)
    path_x = vis_data['rrt_path_x']
    path_y = vis_data['rrt_path_y']
    for i in range(len(path_x) - 1):
        line_id = p.addUserDebugLine(
            [path_x[i], path_y[i], 0.05],
            [path_x[i+1], path_y[i+1], 0.05],
            lineColorRGB=[0, 1, 0],
            lineWidth=2
        )
        new_line_ids.append(line_id)
    
    # Draw waypoints (yellow spheres)
    wp_x = vis_data['waypoint_x']
    wp_y = vis_data['waypoint_y']
    for i in range(len(wp_x)):
        sphere_id = p.addUserDebugPoints(
            [[wp_x[i], wp_y[i], 0.1]],
            [[1, 1, 0]],  # Yellow
            pointSize=10
        )
        new_line_ids.append(sphere_id)
    
    # Highlight current target (red)
    if vis_data['current_target']:
        target = vis_data['current_target']
        target_id = p.addUserDebugPoints(
            [[target[0], target[1], 0.15]],
            [[1, 0, 0]],  # Red
            pointSize=15
        )
        new_line_ids.append(target_id)
    
    return new_line_ids


def save_log(log_data):
    """Save simulation log to JSON file."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = log_data['metadata']['start_time']
    filename = f"mpc_rrt_comparison_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"\n===== MPC+RRT Log Saved =====")
    print(f"File: {filepath}")
    print(f"Total steps: {len(log_data['trajectory'])}")

    if len(log_data['controls']) > 0:
        velocities = [c['state']['v'] for c in log_data['controls']]
        solve_times = [c['solver']['solve_time_ms'] for c in log_data['controls']]

        print(f"\n--- Statistics ---")
        print(f"Velocity: min={min(velocities):.2f}, max={max(velocities):.2f}, avg={np.mean(velocities):.2f} m/s")
        print(f"Solve time: min={min(solve_times):.2f}, max={max(solve_times):.2f}, avg={np.mean(solve_times):.2f} ms")

        solver_failures = sum(1 for c in log_data['controls'] if c['solver']['status'] != 0)
        print(f"Solver failures: {solver_failures}/{len(log_data['controls'])} ({100*solver_failures/len(log_data['controls']):.1f}%)")

    return filepath


def main():
    print("=" * 60)
    print("MPC + RRT* Integrated Comparison Simulation")
    print("=" * 60)
    print(f"Map: {MAP_SIZE}x{MAP_SIZE}")
    print(f"Start: {START_POS}, Goal: {GOAL_POS}")
    print(f"Obstacles: {len(STATIC_OBSTACLES)}")
    print(f"Mode: {'HEADLESS' if HEADLESS else 'GUI'}")
    print("=" * 60)

    # Setup environment
    setup_environment(headless=HEADLESS)
    car_id = create_vehicle()
    obstacle_ids = create_obstacles()
    static_obstacles = get_mpc_obstacles()

    # Initialize car state
    car_state = np.array([START_POS[0], START_POS[1], 1.0, START_YAW])

    # Initialize MPC+RRT controller
    print("\nPlanning RRT* path...")
    planning_start = time.time()
    
    controller = create_mpc_rrt_controller(
        start_state=car_state.tolist(),
        goal=GOAL_POS,
        static_obstacles=static_obstacles,
        map_size=MAP_SIZE,
        horizon=MPC_HORIZON,
        dt=MPC_DT,
        waypoint_tolerance=2.0,
        lookahead_distance=4.0,
        replan_on_deviation=15.0  # High threshold - rely on MPC obstacle avoidance
    )
    
    planning_time = time.time() - planning_start
    
    # Get planning stats
    stats = controller.get_planning_stats()
    print(f"\nRRT* Planning Complete:")
    print(f"  Success: {stats['planning_success']}")
    print(f"  Time: {stats['planning_time']:.3f}s")
    print(f"  Iterations: {stats['iterations_used']}")
    print(f"  Waypoints: {stats['num_waypoints']}")
    print(f"  Path Length: {stats['path_length']:.2f}m")
    
    if not stats['planning_success']:
        print("\n*** RRT Planning Failed! Exiting. ***")
        p.disconnect()
        return
    
    # Visualize path (GUI only)
    path_line_ids = []
    if not HEADLESS:
        path_line_ids = visualize_rrt_path(controller)

    # Initialize log
    simulation_log = {
        'metadata': {
            'start_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'planner': 'MPC+RRT',
            'goal': list(GOAL_POS),
            'start': list(START_POS),
            'map_size': MAP_SIZE,
            'max_speed': MAX_SPEED,
            'min_speed': MIN_SPEED,
            'dt': CONTROLLER_DT,
            'mpc_horizon': MPC_HORIZON,
            'mpc_dt': MPC_DT,
            'static_obstacles': [{'position': obs['position'], 'radius': obs['radius']} for obs in STATIC_OBSTACLES],
            'rrt_planning': {
                'planning_time_s': stats['planning_time'],
                'iterations': stats['iterations_used'],
                'num_waypoints': stats['num_waypoints'],
                'path_length': stats['path_length']
            }
        },
        'waypoints': [(wp.x, wp.y) for wp in controller.waypoints],
        'trajectory': [],
        'controls': []
    }

    print("\nRunning MPC+RRT simulation...")
    simulation_start = time.time()

    for step in range(MAX_STEPS):
        current_time = step * CONTROLLER_DT
        cur_state_np = np.array(car_state)

        # Solve MPC+RRT
        solve_start = time.time()
        optimal_control, solver_status, debug_info = controller.solve(cur_state_np.tolist(), debug=False)
        solve_time = time.time() - solve_start

        # Get solver cost (from MPC)
        try:
            solver_cost = float(controller.mpc.solver.get_cost())
        except:
            solver_cost = -1.0

        # Log current step
        dist_to_goal = math.hypot(car_state[0] - GOAL_POS[0], car_state[1] - GOAL_POS[1])
        min_obs_dist, closest_obs_idx = compute_min_obstacle_distance(car_state[0:2], static_obstacles)

        simulation_log['trajectory'].append({
            'step': step,
            'x': float(car_state[0]),
            'y': float(car_state[1]),
            'v': float(car_state[2]),
            'psi': float(car_state[3])
        })

        simulation_log['controls'].append({
            'step': step,
            'time': current_time,
            'state': {
                'x': float(car_state[0]),
                'y': float(car_state[1]),
                'v': float(car_state[2]),
                'psi': float(car_state[3]),
                'psi_deg': float(np.degrees(car_state[3]))
            },
            'control': {
                'steering': float(optimal_control[0]),
                'steering_deg': float(np.degrees(optimal_control[0])),
                'acceleration': float(optimal_control[1])
            },
            'solver': {
                'status': int(solver_status),
                'solve_time_ms': solve_time * 1000,
                'cost': solver_cost
            },
            'waypoint': {
                'current_idx': debug_info.get('current_waypoint_idx', -1),
                'total': debug_info.get('total_waypoints', 0),
                'target': debug_info.get('target_waypoint', None),
                'deviation': debug_info.get('deviation', 0.0)
            },
            'distances': {
                'to_goal': float(dist_to_goal),
                'to_nearest_obstacle': float(min_obs_dist),
                'collision': min_obs_dist < 0
            }
        })

        # Check goal reached (from controller)
        if debug_info.get('goal_reached', False):
            print(f"*** Goal reached at step {step}! ***")
            break

        # Update vehicle state
        car_state = vehicle_simulate_engine(car_state, optimal_control, CONTROLLER_DT)

        # Update PyBullet visualization
        car_pos = [car_state[0], car_state[1], 0.1]
        car_orn = p.getQuaternionFromEuler([0, 0, car_state[3]])
        p.resetBasePositionAndOrientation(car_id, car_pos, car_orn)

        # Update path visualization (GUI only, every 20 steps)
        if not HEADLESS and step % 20 == 0:
            path_line_ids = visualize_rrt_path(controller, path_line_ids)

        # Collision check
        collided = False
        for obs_id in obstacle_ids:
            if len(p.getContactPoints(car_id, obs_id)) > 0:
                collided = True
                break

        simulation_log['controls'][-1]['distances']['collision'] = collided
        
        # Print collision details if collision detected
        if collided and closest_obs_idx >= 0:
            obs = static_obstacles[closest_obs_idx]
            print(f"  [COLLISION] step={step} car=({car_state[0]:.2f}, {car_state[1]:.2f}) v={car_state[2]:.2f} "
                  f"obs{closest_obs_idx}=({obs['position'][0]:.1f}, {obs['position'][1]:.1f}) "
                  f"clearance={min_obs_dist:.3f}m")

        # Goal check (backup)
        if dist_to_goal < GOAL_TOLERANCE:
            print(f"*** Goal reached at step {step}! ***")
            break

        p.stepSimulation()
        if not HEADLESS:
            time.sleep(0.01)

    simulation_time = time.time() - simulation_start

    # Compute path length
    path_length = 0.0
    for i in range(len(simulation_log['trajectory']) - 1):
        dx = simulation_log['trajectory'][i+1]['x'] - simulation_log['trajectory'][i]['x']
        dy = simulation_log['trajectory'][i+1]['y'] - simulation_log['trajectory'][i]['y']
        path_length += math.hypot(dx, dy)

    straight_line_dist = math.hypot(GOAL_POS[0] - START_POS[0], GOAL_POS[1] - START_POS[1])

    # Summary
    min_clearance = min([c['distances']['to_nearest_obstacle'] for c in simulation_log['controls']])
    collisions = sum(1 for c in simulation_log['controls'] if c['distances']['collision'])
    total_solve_time = sum([c['solver']['solve_time_ms'] for c in simulation_log['controls']])
    solver_failures = sum(1 for c in simulation_log['controls'] if c['solver']['status'] != 0)

    simulation_log['summary'] = {
        'total_steps': len(simulation_log['trajectory']),
        'total_time_s': len(simulation_log['trajectory']) * CONTROLLER_DT,
        'path_length': path_length,
        'straight_line_distance': straight_line_dist,
        'path_efficiency': straight_line_dist / path_length if path_length > 0 else 0,
        'final_distance_to_goal': simulation_log['controls'][-1]['distances']['to_goal'],
        'min_obstacle_clearance': min_clearance,
        'collision_count': collisions,
        'solver_failure_count': solver_failures,
        'total_solve_time_ms': total_solve_time,
        'avg_solve_time_ms': total_solve_time / len(simulation_log['controls']) if simulation_log['controls'] else 0,
        'rrt_planning_time_s': stats['planning_time'],
        'simulation_wall_time_s': simulation_time
    }

    # Save log
    save_log(simulation_log)

    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
    print(f"Path length: {path_length:.2f} m")
    print(f"Path efficiency: {simulation_log['summary']['path_efficiency']*100:.1f}%")
    print(f"Final distance to goal: {simulation_log['summary']['final_distance_to_goal']:.2f} m")
    print(f"Min obstacle clearance: {min_clearance:.2f} m")
    print(f"Collisions: {collisions}")
    print(f"Solver failures: {solver_failures}/{len(simulation_log['controls'])} ({100*solver_failures/len(simulation_log['controls']):.1f}%)")
    print(f"RRT planning time: {stats['planning_time']:.3f}s")
    print(f"Total MPC solve time: {total_solve_time:.2f} ms")
    print("=" * 60)

    if HEADLESS:
        p.disconnect()
    else:
        # Keep window open
        print("\nPress Ctrl+C to exit...")
        try:
            while True:
                p.stepSimulation()
                time.sleep(0.01)
        except KeyboardInterrupt:
            p.disconnect()


if __name__ == "__main__":
    main()
