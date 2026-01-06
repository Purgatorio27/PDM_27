import sys
import os
import pybullet as p
import time
import numpy as np
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from Environment.environment import MazeEnvironment
except ModuleNotFoundError:
    sys.path.append(os.path.join(root_dir, "Environment"))
    from environment import MazeEnvironment
    
from KinematicBicycleModelRRT import KinematicBicycleModelRRT, State
from RRTStar import RRTStar 
from utils import generate_trajectory, save_simulation_log

# All parameters
DIFFICULTY = "Advanced"  # Choose: "Simple", "Intermediate", "Advanced" ("Expert" and "Master" exist but are too difficult)

MIN_TURN_RAD = 0.8
TARGET_VELOCITY = 5.0  # m/s

MAX_ITERATIONS = 20000
MAX_ATTEMPTS = 5

SMOOTH = True
SMOOTHING_FACTOR = 30

SIM_SLEEP_TIME = 0.0001  # tried to make the simulation faster but no difference
SPEED_MULTIPLIER = 5  # skipping 5 frames for speed


def main_RRT():
    env = MazeEnvironment(difficulty=DIFFICULTY)
    model = KinematicBicycleModelRRT(min_radius=MIN_TURN_RAD)
    
    START_POSE = env.config['start']
    GOAL_POS = env.config['goal']
    
    start_state = State(START_POSE[0], START_POSE[1], START_POSE[2])
    
    simulation_log = {
        'metadata': {
            'start_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'computation_time': 0.0,
            'travel_time': 0.0,
            'start_pose': START_POSE,
            'goal_pos': GOAL_POS,
            'parameters': {
                'min_turn_rad': MIN_TURN_RAD,
                'target_velocity': TARGET_VELOCITY,
                'max_iterations': MAX_ITERATIONS,
                'difficulty': DIFFICULTY
            }
        },
        'planned_rrt_path': [],
        'executed_trajectory': []
    }

    vehicle = env.spawn_car(START_POSE)
    env.draw_goal(GOAL_POS)

    planner = RRTStar(start_state, GOAL_POS, model, env)
    print(f"Starting RRT* Planning: {DIFFICULTY} Maze")
    solve_start = time.time()
    
    RRT_trajectory, trajectory = generate_trajectory(
        planner, SMOOTHING_FACTOR, TARGET_VELOCITY, MAX_ATTEMPTS, MAX_ITERATIONS
    )

    solve_end = time.time()
    simulation_log['metadata']['computation_time'] = solve_end - solve_start

    if trajectory is not None:
        print(f"Path found in {solve_end - solve_start:.2f} seconds")

        diffs = np.diff(trajectory[:, :2], axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        total_distance = np.sum(distances)
        
        print(f"Total distance to travel: {total_distance:.2f} meters")
        
        simulation_log['metadata']['total_distance'] = float(total_distance)

        for i in range(len(RRT_trajectory) - 1):
            p.addUserDebugLine([RRT_trajectory[i].x, RRT_trajectory[i].y, 0.1], 
                               [RRT_trajectory[i+1].x, RRT_trajectory[i+1].y, 0.1], [0, 0, 1], 2)

        if SMOOTH:
            for i in range(len(trajectory) - 1):
                p.addUserDebugLine([trajectory[i, 0], trajectory[i, 1], 0.12], 
                                   [trajectory[i+1, 0], trajectory[i+1, 1], 0.12], [1, 0, 0], 4)

        print("Executing trajectory")
        travel_start = time.time()
        
        for i in range(0, len(trajectory), SPEED_MULTIPLIER):
            target_x, target_y, target_yaw, target_v = trajectory[i]
            
            p.resetBasePositionAndOrientation(
                vehicle, [target_x, target_y, 0.15], p.getQuaternionFromEuler([0, 0, target_yaw])
            )

            simulation_log['executed_trajectory'].append({
                'step': i, 
                'x': float(target_x), 
                'y': float(target_y),
                'yaw': float(target_yaw), 
                'v': float(target_v)
            })
            
            p.stepSimulation()
            time.sleep(SIM_SLEEP_TIME)

        travel_end = time.time()
        simulation_log['metadata']['travel_time'] = travel_end - travel_start

        method_name = "RRTStar"
        log_dir = os.path.join(root_dir, "logs", DIFFICULTY, method_name)
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(log_dir, f"log_{timestamp}.json")
        
        save_simulation_log(simulation_log, save_path)

        print("Simulation finished")
    else:
        print(f"RRT* could not find a path in {DIFFICULTY} maze")

    # Keep PyBullet window open
    while p.isConnected():
        p.stepSimulation()
        time.sleep(0.01)

if __name__ == "__main__":
    main_RRT()