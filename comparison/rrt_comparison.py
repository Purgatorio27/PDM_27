"""
RRT* simulation for comparison with MPC.
Uses shared_config.py for identical environment setup.
Includes JSON logging matching MPC format.

Usage:
    python rrt_comparison.py          # GUI mode
    python rrt_comparison.py --headless  # Headless mode (for testing)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RRTStar'))

# Check for headless mode
HEADLESS = '--headless' in sys.argv

import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import json
from datetime import datetime
from scipy.interpolate import make_interp_spline

from shared_config import (
    MAP_SIZE, START_POS, GOAL_POS, START_YAW,
    STATIC_OBSTACLES, VEHICLE_RADIUS, GOAL_TOLERANCE,
    MAX_SPEED, MIN_SPEED
)
from RRTStar.KinematicBicycleModelRRT import KinematicBicycleModelRRT, State


# ============== RRT* Parameters ==============
MIN_TURN_RAD = 0.8
MAX_ITERATIONS = 20000
MAX_ATTEMPTS = 5
STEP_SIZE = 0.8  # Larger step for bigger map
SEARCH_RADIUS = 3.0
GOAL_BIAS = 0.15
SMOOTHING_FACTOR = 30
SIM_SLEEP_TIME = 0.02


class ComparisonEnvironment:
    """Environment for 25x25 map with circle obstacles from shared config."""

    def __init__(self, headless=False):
        if not p.isConnected():
            if headless:
                p.connect(p.DIRECT)
            else:
                p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        # Camera for 25x25 map (only in GUI mode)
        if not headless:
            p.resetDebugVisualizerCamera(
                cameraDistance=20,
                cameraYaw=0,
                cameraPitch=-89.9,
                cameraTargetPosition=[MAP_SIZE/2, MAP_SIZE/2, 0]
            )

        self.obstacles = []
        self._build_world()

    def _build_world(self):
        """Create walls and obstacles from shared config."""
        # Boundary walls
        wall_height = 1.0
        wall_thickness = 0.2
        wall_color = [0.6, 0.6, 0.6, 1]

        walls = [
            {'pos': [MAP_SIZE/2, 0], 'dim': [MAP_SIZE/2, wall_thickness]},  # Bottom
            {'pos': [MAP_SIZE/2, MAP_SIZE], 'dim': [MAP_SIZE/2, wall_thickness]},  # Top
            {'pos': [0, MAP_SIZE/2], 'dim': [wall_thickness, MAP_SIZE/2]},  # Left
            {'pos': [MAP_SIZE, MAP_SIZE/2], 'dim': [wall_thickness, MAP_SIZE/2]},  # Right
        ]

        for wall in walls:
            half_ext = [wall['dim'][0], wall['dim'][1], wall_height/2]
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=wall_color),
                basePosition=[wall['pos'][0], wall['pos'][1], wall_height/2]
            )

        # Circle obstacles from shared config
        for obs in STATIC_OBSTACLES:
            pos = obs['position']
            radius = obs['radius']

            # Store for collision checking
            self.obstacles.append({
                'type': 'circle',
                'pos': pos,
                'radius': radius
            })

            # Visual/collision in PyBullet
            col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=1.0)
            vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=1.0, rgbaColor=[0, 0, 1, 0.8])
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[pos[0], pos[1], 0.5]
            )

        # Goal marker
        goal_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 0, 0.8])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_shape, basePosition=[GOAL_POS[0], GOAL_POS[1], 0.1])

    def spawn_car(self, start_state):
        """Spawn vehicle at start position."""
        return p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.1]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.1], rgbaColor=[1, 0, 0, 1]),
            basePosition=[start_state.x, start_state.y, 0.15],
            baseOrientation=p.getQuaternionFromEuler([0, 0, start_state.yaw])
        )


class RRTStarComparison:
    """RRT* planner adapted for 25x25 map with circle obstacles."""

    def __init__(self, start, goal, model, env):
        self.nodes = [start]
        self.goal_pos = goal
        self.model = model
        self.env = env
        self.parents = {0: None}
        self.step_size = STEP_SIZE
        self.search_radius = SEARCH_RADIUS
        self.car_radius = VEHICLE_RADIUS + 0.2  # Safety margin

        # Planning metrics
        self.iterations_used = 0
        self.planning_time = 0

    def collision_checker(self, x, y):
        """Check if position is collision-free."""
        # Boundary check
        margin = 0.5
        if x < margin or x > MAP_SIZE - margin or y < margin or y > MAP_SIZE - margin:
            return False

        # Circle obstacle check
        for obs in self.env.obstacles:
            ox, oy = obs['pos']
            obs_radius = obs['radius']
            dist = math.hypot(x - ox, y - oy)
            if dist < (obs_radius + self.car_radius):
                return False
        return True

    def plan(self, max_iter=MAX_ITERATIONS):
        """Plan path using RRT*."""
        start_time = time.time()

        for i in range(max_iter):
            self.iterations_used = i + 1

            # Sample with goal bias
            if np.random.rand() < GOAL_BIAS:
                sample = self.goal_pos
            else:
                sample = (
                    np.random.uniform(0.5, MAP_SIZE - 0.5),
                    np.random.uniform(0.5, MAP_SIZE - 0.5)
                )

            # Find nearest node
            dists = [math.hypot(n.x - sample[0], n.y - sample[1]) for n in self.nodes]
            nearest_idx = np.argmin(dists)

            # Extend toward sample
            new_node = self.model.next_state(self.nodes[nearest_idx], sample, self.step_size)

            if new_node and self.collision_checker(new_node.x, new_node.y):
                # Find nearby nodes for rewiring
                nearby = [j for j, n in enumerate(self.nodes)
                         if math.hypot(n.x - new_node.x, n.y - new_node.y) < self.search_radius]

                best_idx = nearest_idx
                min_cost = self.nodes[nearest_idx].cost + self.step_size

                # Choose best parent
                for neighbor_idx in nearby:
                    reached = self.model.next_state(self.nodes[neighbor_idx], (new_node.x, new_node.y), self.step_size)
                    if reached and math.hypot(reached.x - new_node.x, reached.y - new_node.y) < 0.1:
                        new_cost = self.nodes[neighbor_idx].cost + self.step_size
                        if new_cost < min_cost:
                            best_idx = neighbor_idx
                            min_cost = new_cost

                # Add to tree
                new_node.cost = min_cost
                self.nodes.append(new_node)
                self.parents[len(self.nodes) - 1] = best_idx

        self.planning_time = time.time() - start_time

        # Find path to goal
        goal_dists = [math.hypot(n.x - self.goal_pos[0], n.y - self.goal_pos[1]) for n in self.nodes]
        min_dist = min(goal_dists)

        if min_dist < 2.0:  # Goal threshold
            return self._extract_path(np.argmin(goal_dists))
        return None

    def _extract_path(self, idx):
        """Extract path from tree."""
        path = []
        while idx is not None:
            path.append(self.nodes[idx])
            idx = self.parents[idx]
        return path[::-1]


def smooth_trajectory(rrt_path, smoothing_factor, target_velocity):
    """Convert RRT path to smooth trajectory with velocity."""
    points = np.array([[n.x, n.y] for n in rrt_path])

    t = np.linspace(0, 1, len(points))
    t_smooth = np.linspace(0, 1, len(points) * smoothing_factor)

    # Cubic spline interpolation
    spline_x = make_interp_spline(t, points[:, 0], k=3)(t_smooth)
    spline_y = make_interp_spline(t, points[:, 1], k=3)(t_smooth)

    # Build trajectory with heading and velocity
    trajectory = []
    for i in range(len(spline_x)):
        if i < len(spline_x) - 1:
            dx = spline_x[i+1] - spline_x[i]
            dy = spline_y[i+1] - spline_y[i]
            yaw = math.atan2(dy, dx)
        else:
            yaw = trajectory[-1][2]
        trajectory.append([spline_x[i], spline_y[i], yaw, target_velocity])

    return np.array(trajectory)


def compute_path_length(trajectory):
    """Compute total path length."""
    length = 0.0
    for i in range(len(trajectory) - 1):
        dx = trajectory[i+1, 0] - trajectory[i, 0]
        dy = trajectory[i+1, 1] - trajectory[i, 1]
        length += math.hypot(dx, dy)
    return length


def compute_min_obstacle_distance(pos, obstacles):
    """Compute minimum distance to any obstacle."""
    min_dist = float('inf')
    for obs in obstacles:
        ox, oy = obs['pos']
        dist = math.hypot(pos[0] - ox, pos[1] - oy) - obs['radius'] - VEHICLE_RADIUS
        min_dist = min(min_dist, dist)
    return min_dist


def save_log(log_data):
    """Save simulation log to JSON file."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = log_data['metadata']['start_time']
    filename = f"rrt_comparison_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"\n===== RRT* Log Saved =====")
    print(f"File: {filepath}")
    print(f"Planning iterations: {log_data['planning_info']['iterations']}")
    print(f"Planning time: {log_data['planning_info']['planning_time_ms']:.2f} ms")
    print(f"Path length: {log_data['planning_info']['path_length']:.2f} m")
    print(f"Trajectory points: {len(log_data['trajectory'])}")

    return filepath


def main():
    print("=" * 50)
    print("RRT* Comparison Simulation")
    print(f"Map: {MAP_SIZE}x{MAP_SIZE}")
    print(f"Start: {START_POS}, Goal: {GOAL_POS}")
    print(f"Obstacles: {len(STATIC_OBSTACLES)}")
    print(f"Mode: {'HEADLESS' if HEADLESS else 'GUI'}")
    print("=" * 50)

    # Initialize environment
    env = ComparisonEnvironment(headless=HEADLESS)
    model = KinematicBicycleModelRRT(min_radius=MIN_TURN_RAD)

    start_state = State(START_POS[0], START_POS[1], START_YAW)
    vehicle = env.spawn_car(start_state)

    # Initialize log
    simulation_log = {
        'metadata': {
            'start_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'planner': 'RRT*',
            'goal': list(GOAL_POS),
            'start': list(START_POS),
            'map_size': MAP_SIZE,
            'max_speed': MAX_SPEED,
            'min_speed': MIN_SPEED,
            'static_obstacles': [{'position': obs['position'], 'radius': obs['radius']} for obs in STATIC_OBSTACLES]
        },
        'planning_info': {},
        'trajectory': [],
        'controls': []
    }

    # Plan path
    print("\nPlanning path with RRT*...")
    planner = RRTStarComparison(start_state, GOAL_POS, model, env)

    rrt_path = None
    for attempt in range(MAX_ATTEMPTS):
        print(f"  Attempt {attempt + 1}/{MAX_ATTEMPTS}...")
        rrt_path = planner.plan(max_iter=MAX_ITERATIONS)
        if rrt_path is not None:
            print(f"  Path found with {len(rrt_path)} waypoints!")
            break

    if rrt_path is None:
        print("ERROR: RRT* could not find a path!")
        simulation_log['planning_info'] = {
            'success': False,
            'iterations': planner.iterations_used,
            'planning_time_ms': planner.planning_time * 1000
        }
        save_log(simulation_log)
        return

    # Smooth trajectory
    trajectory = smooth_trajectory(rrt_path, SMOOTHING_FACTOR, MAX_SPEED * 0.5)
    path_length = compute_path_length(trajectory)
    straight_line_dist = math.hypot(GOAL_POS[0] - START_POS[0], GOAL_POS[1] - START_POS[1])

    simulation_log['planning_info'] = {
        'success': True,
        'iterations': planner.iterations_used,
        'planning_time_ms': planner.planning_time * 1000,
        'rrt_waypoints': len(rrt_path),
        'smoothed_points': len(trajectory),
        'path_length': path_length,
        'straight_line_distance': straight_line_dist,
        'path_efficiency': straight_line_dist / path_length if path_length > 0 else 0
    }

    print(f"\nPath statistics:")
    print(f"  Path length: {path_length:.2f} m")
    print(f"  Straight-line: {straight_line_dist:.2f} m")
    print(f"  Efficiency: {simulation_log['planning_info']['path_efficiency']*100:.1f}%")

    # Visualize paths (GUI mode only)
    if not HEADLESS:
        for i in range(len(rrt_path) - 1):
            p.addUserDebugLine(
                [rrt_path[i].x, rrt_path[i].y, 0.1],
                [rrt_path[i+1].x, rrt_path[i+1].y, 0.1],
                [0, 0, 1], 2
            )
        for i in range(0, len(trajectory) - 1, 5):
            p.addUserDebugLine(
                [trajectory[i, 0], trajectory[i, 1], 0.12],
                [trajectory[i+1, 0], trajectory[i+1, 1], 0.12],
                [1, 0, 0], 1
            )

    # Execute trajectory
    print("\nExecuting trajectory...")
    execution_start = time.time()

    for step, point in enumerate(trajectory):
        target_x, target_y, target_yaw, target_v = point

        # Update vehicle position
        p.resetBasePositionAndOrientation(
            vehicle,
            [target_x, target_y, 0.15],
            p.getQuaternionFromEuler([0, 0, target_yaw])
        )

        # Compute distances
        dist_to_goal = math.hypot(target_x - GOAL_POS[0], target_y - GOAL_POS[1])
        min_obs_dist = compute_min_obstacle_distance([target_x, target_y], env.obstacles)

        # Log trajectory point
        simulation_log['trajectory'].append({
            'step': step,
            'x': float(target_x),
            'y': float(target_y),
            'v': float(target_v),
            'psi': float(target_yaw)
        })

        # Log control entry (for compatibility with MPC format)
        simulation_log['controls'].append({
            'step': step,
            'time': step * SIM_SLEEP_TIME,
            'state': {
                'x': float(target_x),
                'y': float(target_y),
                'v': float(target_v),
                'psi': float(target_yaw),
                'psi_deg': float(np.degrees(target_yaw))
            },
            'distances': {
                'to_goal': float(dist_to_goal),
                'to_nearest_obstacle': float(min_obs_dist),
                'collision': min_obs_dist < 0
            }
        })

        p.stepSimulation()
        if not HEADLESS:
            time.sleep(SIM_SLEEP_TIME)

        # Check goal reached
        if dist_to_goal < GOAL_TOLERANCE:
            print(f"Goal reached at step {step}!")
            break

    execution_time = time.time() - execution_start
    simulation_log['planning_info']['execution_time_s'] = execution_time

    # Compute final statistics
    min_clearance = min([c['distances']['to_nearest_obstacle'] for c in simulation_log['controls']])
    collisions = sum(1 for c in simulation_log['controls'] if c['distances']['collision'])

    simulation_log['summary'] = {
        'total_steps': len(simulation_log['trajectory']),
        'total_time_s': len(simulation_log['trajectory']) * SIM_SLEEP_TIME,
        'final_distance_to_goal': simulation_log['controls'][-1]['distances']['to_goal'],
        'min_obstacle_clearance': min_clearance,
        'collision_count': collisions,
        'total_planning_time_ms': simulation_log['planning_info']['planning_time_ms'],
        'total_execution_time_s': execution_time
    }

    # Save log
    save_log(simulation_log)

    print("\n===== Simulation Complete =====")
    print(f"Final distance to goal: {simulation_log['summary']['final_distance_to_goal']:.2f} m")
    print(f"Min obstacle clearance: {min_clearance:.2f} m")
    print(f"Collisions: {collisions}")

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
