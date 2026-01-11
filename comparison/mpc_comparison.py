"""
MPC simulation for comparison with RRT*.
Uses shared_config.py for identical environment setup.
Static obstacles only - no dynamic obstacles.

Usage:
    python mpc_comparison.py          # GUI mode
    python mpc_comparison.py --headless  # Headless mode (for testing)
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

# Patch MPC Config before importing MPC_core
import MPC.Config as mpc_cfg
mpc_cfg.goal = GOAL_POS
mpc_cfg.vehicle_length = VEHICLE_LENGTH
mpc_cfg.vehicle_width = VEHICLE_WIDTH
mpc_cfg.vehicle_radius = VEHICLE_RADIUS
mpc_cfg.lf = VEHICLE_LENGTH * 0.5
mpc_cfg.lr = VEHICLE_LENGTH * 0.5

# Import MPC controller (after config patch)
from MPC.MPC_core import MPC


# ============== MPC Parameters ==============
CONTROLLER_DT = 0.05  # 20Hz control
MAX_STEPS = 1200  # Max simulation steps (Increased for 40x40 map)
MPC_HORIZON = 40
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
    for obs in obstacles:
        ox, oy = obs['position']
        dist = math.hypot(car_pos[0] - ox, car_pos[1] - oy) - obs['radius'] - VEHICLE_RADIUS
        min_dist = min(min_dist, dist)
    return min_dist


def compute_emergency_avoidance(car_state, obstacles, goal_pos, recovery_step, total_recovery_steps):
    """
    Compute emergency avoidance control when deadlock detected.
    Strategy:
    1. First 40% of duration: Reverse blindly to clear local minimum.
    2. Remaining 60%: Turn aggressively towards the clearest path (tangent to obstacle).
    """
    x, y, v, psi = car_state
    
    # 1. Reverse Phase
    if recovery_step < total_recovery_steps * 0.4:
        # Reverse straight
        return np.array([0.0, -1.0])

    # 2. Forward Escape Phase
    min_dist = float('inf')
    nearest_obs_pos = None

    for obs in obstacles:
        ox, oy = obs['position']
        # Use collision radius + small buffer to determine nearest
        dist = math.hypot(x - ox, y - oy) - obs['radius'] - VEHICLE_RADIUS
        if dist < min_dist:
            min_dist = dist
            nearest_obs_pos = (ox, oy)
    
    # If we are in forward phase but dangerously close to an obstacle, reverse instead
    if min_dist < 0.1: # Less than 10cm clearance
         return np.array([0.0, -1.0])

    if nearest_obs_pos is None:
        goal_angle = math.atan2(goal_pos[1] - y, goal_pos[0] - x)
        heading_error = goal_angle - psi
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        steer = np.clip(heading_error * 1.5, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
        return np.array([steer, 1.5])

    ox, oy = nearest_obs_pos
    dx = x - ox
    dy = y - oy
    angle_from_obs = math.atan2(dy, dx)

    goal_angle = math.atan2(goal_pos[1] - y, goal_pos[0] - x)

    # Tangent angles (90 degrees to obstacle direction)
    perp_angle1 = angle_from_obs + math.pi / 2
    perp_angle2 = angle_from_obs - math.pi / 2

    # Choose tangent closer to goal
    diff1 = abs(math.atan2(math.sin(goal_angle - perp_angle1), math.cos(goal_angle - perp_angle1)))
    diff2 = abs(math.atan2(math.sin(goal_angle - perp_angle2), math.cos(goal_angle - perp_angle2)))

    target_angle = perp_angle1 if diff1 < diff2 else perp_angle2
    heading_error = target_angle - psi
    heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

    # Aggressive steering
    steer = np.clip(heading_error * 3.0, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    
    # Brake if heading error is large to turn in place
    if abs(heading_error) > np.radians(30):
        accel = 0.5 # Slow forward
    else:
        accel = 1.0

    return np.array([steer, accel])


def save_log(log_data):
    """Save simulation log to JSON file."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = log_data['metadata']['start_time']
    filename = f"mpc_comparison_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"\n===== MPC Log Saved =====")
    print(f"File: {filepath}")
    print(f"Total steps: {len(log_data['trajectory'])}")

    if len(log_data['controls']) > 0:
        velocities = [c['state']['v'] for c in log_data['controls']]
        costs = [c['solver']['cost'] for c in log_data['controls'] if c['solver']['cost'] > 0]
        solve_times = [c['solver']['solve_time_ms'] for c in log_data['controls']]

        print(f"\n--- Statistics ---")
        print(f"Velocity: min={min(velocities):.2f}, max={max(velocities):.2f}, avg={np.mean(velocities):.2f} m/s")
        if costs:
            print(f"Cost: min={min(costs):.2f}, max={max(costs):.2f}, avg={np.mean(costs):.2f}")
        print(f"Solve time: min={min(solve_times):.2f}, max={max(solve_times):.2f}, avg={np.mean(solve_times):.2f} ms")

        solver_failures = sum(1 for c in log_data['controls'] if c['solver']['status'] != 0)
        print(f"Solver failures: {solver_failures}/{len(log_data['controls'])} ({100*solver_failures/len(log_data['controls']):.1f}%)")

    return filepath


def main():
    print("=" * 50)
    print("MPC Comparison Simulation")
    print(f"Map: {MAP_SIZE}x{MAP_SIZE}")
    print(f"Start: {START_POS}, Goal: {GOAL_POS}")
    print(f"Obstacles: {len(STATIC_OBSTACLES)}")
    print(f"Mode: {'HEADLESS' if HEADLESS else 'GUI'}")
    print("=" * 50)

    # Setup environment
    setup_environment(headless=HEADLESS)
    car_id = create_vehicle()
    obstacle_ids = create_obstacles()
    static_obstacles = get_mpc_obstacles()

    # Initialize MPC controller (no dynamic obstacles, with explicit goal)
    mpc_controller = MPC(static_obstacles, [], horizon=MPC_HORIZON, dt=MPC_DT, goal=GOAL_POS)

    # Initialize car state
    car_state = np.array([START_POS[0], START_POS[1], 1.0, START_YAW])

    # Initialize log
    simulation_log = {
        'metadata': {
            'start_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'planner': 'MPC',
            'goal': list(GOAL_POS),
            'start': list(START_POS),
            'map_size': MAP_SIZE,
            'max_speed': MAX_SPEED,
            'min_speed': MIN_SPEED,
            'dt': CONTROLLER_DT,
            'mpc_horizon': MPC_HORIZON,
            'mpc_dt': MPC_DT,
            'static_obstacles': [{'position': obs['position'], 'radius': obs['radius']} for obs in STATIC_OBSTACLES]
        },
        'trajectory': [],
        'mpc_predictions': [],
        'controls': []
    }

    # Deadlock detection
    low_speed_counter = 0
    LOW_SPEED_THRESHOLD = 0.3
    DEADLOCK_STEPS = 20
    last_position = np.array([START_POS[0], START_POS[1]])
    position_change_threshold = 0.1
    deadlock_recovery_mode = False
    recovery_steps = 0
    RECOVERY_DURATION = 60
    
    # Stagnation detection
    best_dist_to_goal = float('inf')
    steps_without_progress = 0
    STAGNATION_STEPS = 100

    print("\nRunning MPC simulation...")
    simulation_start = time.time()

    for step in range(MAX_STEPS):
        current_time = step * CONTROLLER_DT
        cur_state_np = np.array(car_state)

        # Solve MPC
        solve_start = time.time()
        optimal_control, solver_status, debug_info = mpc_controller.solve(cur_state_np, debug=False)
        solve_time = time.time() - solve_start

        # Get MPC predicted trajectory
        mpc_predicted_traj = []
        try:
            for k in range(mpc_controller.N + 1):
                pred_state = mpc_controller.solver.get(k, 'x')
                mpc_predicted_traj.append(pred_state.tolist())
        except:
            pass

        # Get solver cost
        try:
            solver_cost = float(mpc_controller.solver.get_cost())
        except:
            solver_cost = -1.0

        # Deadlock & Stagnation Detection
        current_speed = abs(car_state[2])
        position_change = np.linalg.norm(car_state[0:2] - last_position)
        current_dist_to_goal = math.hypot(car_state[0] - GOAL_POS[0], car_state[1] - GOAL_POS[1])

        # Emergency Braking (Safety Shield)
        # If MPC fails to keep safe distance, override control
        min_dist_check = compute_min_obstacle_distance(car_state[0:2], static_obstacles)
        if min_dist_check < 0.2 and not deadlock_recovery_mode:
             # Too close! Brake/Reverse hard.
             if current_speed > 0.1:
                 optimal_control = np.array([0.0, -1.0 * MAX_DECELERATION]) # Max Brake
             else:
                 optimal_control = np.array([0.0, -1.0]) # Reverse
             pass
             # We rely on the control override.

        # 1. Low Speed Deadlock
        if current_speed < LOW_SPEED_THRESHOLD and position_change < position_change_threshold:
            low_speed_counter += 1
        else:
            low_speed_counter = 0
            if not deadlock_recovery_mode:
                last_position = car_state[0:2].copy()

        # 2. Stagnation Detection
        if current_dist_to_goal < best_dist_to_goal - 0.5:
            best_dist_to_goal = current_dist_to_goal
            steps_without_progress = 0
        else:
            steps_without_progress += 1

        trigger_recovery = (low_speed_counter >= DEADLOCK_STEPS) or (steps_without_progress >= STAGNATION_STEPS)

        if trigger_recovery and not deadlock_recovery_mode:
            deadlock_recovery_mode = True
            recovery_steps = 0
            reason = "Low Speed" if low_speed_counter >= DEADLOCK_STEPS else "Stagnation"
            print(f"[Step {step}] *** DEADLOCK DETECTED ({reason})! ***")
            steps_without_progress = 0 # Reset to give chance after recovery

        if deadlock_recovery_mode:
            recovery_steps += 1
            optimal_control = compute_emergency_avoidance(car_state, static_obstacles, GOAL_POS, recovery_steps, RECOVERY_DURATION)
            # Only exit recovery if we've completed the sequence or made VERY significant progress away
            if recovery_steps >= RECOVERY_DURATION:
                deadlock_recovery_mode = False
                low_speed_counter = 0
                print(f"[Step {step}] Recovery mode ended (Timeout)")
            elif recovery_steps > RECOVERY_DURATION * 0.5 and current_speed > 1.0 and position_change > 0.5:
                 # Early exit only allowed in forward phase
                deadlock_recovery_mode = False
                low_speed_counter = 0
                print(f"[Step {step}] Recovery mode ended (Progress made)")

        last_position = car_state[0:2].copy()

        # Log current step
        dist_to_goal = math.hypot(car_state[0] - GOAL_POS[0], car_state[1] - GOAL_POS[1])
        min_obs_dist = compute_min_obstacle_distance(car_state[0:2], static_obstacles)

        simulation_log['trajectory'].append({
            'step': step,
            'x': float(car_state[0]),
            'y': float(car_state[1]),
            'v': float(car_state[2]),
            'psi': float(car_state[3])
        })

        if len(mpc_predicted_traj) > 0:
            simulation_log['mpc_predictions'].append({
                'step': step,
                'predicted_trajectory': mpc_predicted_traj
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
            'distances': {
                'to_goal': float(dist_to_goal),
                'to_nearest_obstacle': float(min_obs_dist),
                'collision': min_obs_dist < 0
            },
            'deadlock_recovery': deadlock_recovery_mode
        })

        # Update vehicle state
        car_state = vehicle_simulate_engine(car_state, optimal_control, CONTROLLER_DT)

        # Update PyBullet visualization
        car_pos = [car_state[0], car_state[1], 0.1]
        car_orn = p.getQuaternionFromEuler([0, 0, car_state[3]])
        p.resetBasePositionAndOrientation(car_id, car_pos, car_orn)

        # Collision check
        collided = False
        for obs_id in obstacle_ids:
            if len(p.getContactPoints(car_id, obs_id)) > 0:
                collided = True
                break

        simulation_log['controls'][-1]['distances']['collision'] = collided

        # Goal check
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

    simulation_log['summary'] = {
        'total_steps': len(simulation_log['trajectory']),
        'total_time_s': len(simulation_log['trajectory']) * CONTROLLER_DT,
        'path_length': path_length,
        'straight_line_distance': straight_line_dist,
        'path_efficiency': straight_line_dist / path_length if path_length > 0 else 0,
        'final_distance_to_goal': simulation_log['controls'][-1]['distances']['to_goal'],
        'min_obstacle_clearance': min_clearance,
        'collision_count': collisions,
        'total_solve_time_ms': total_solve_time,
        'avg_solve_time_ms': total_solve_time / len(simulation_log['controls']) if simulation_log['controls'] else 0,
        'simulation_wall_time_s': simulation_time
    }

    # Save log
    save_log(simulation_log)

    print("\n===== Simulation Complete =====")
    print(f"Path length: {path_length:.2f} m")
    print(f"Path efficiency: {simulation_log['summary']['path_efficiency']*100:.1f}%")
    print(f"Final distance to goal: {simulation_log['summary']['final_distance_to_goal']:.2f} m")
    print(f"Min obstacle clearance: {min_clearance:.2f} m")
    print(f"Collisions: {collisions}")
    print(f"Total MPC solve time: {total_solve_time:.2f} ms")

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
