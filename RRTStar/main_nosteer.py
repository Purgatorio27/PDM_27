import pybullet as p
import time
from datetime import datetime

from KinematicBicycleModelRRT import State, SimpleHolonomicModel
from Environment import Environment
from RRTStar import RRTStar
from utils import generate_trajectory, save_simulation_log


# Parameters
USE_RANDOM_ENV = False
MIN_TURN_RAD = 0.0 

START_POSE = (-4.2, -4.2, 0.0) 
GOAL_POS = (4.2, 4.2)

MAX_ITERATIONS = 10000
MAX_ATTEMPTS = 5

SMOOTH = False
SMOOTHING_FACTOR = 1

TARGET_VELOCITY = 5.0
SIM_SLEEP_TIME = 0.05 


def main_RRT():
    env = Environment(randomize=USE_RANDOM_ENV)
    model = SimpleHolonomicModel()
    
    start_state = State(START_POSE[0], START_POSE[1], START_POSE[2])
    
    simulation_log = {
        'metadata': {
            'start_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'computation_time': 0.0,
            'travel_time': 0.0,
            'start_pose': START_POSE,
            'goal_pos': GOAL_POS,
            'parameters': {
                'model_type': 'holonomic',
                'target_velocity': TARGET_VELOCITY,
                'max_iterations': MAX_ITERATIONS
            }
        },
        'planned_rrt_path': [],
        'executed_trajectory': []
    }

    vehicle = env.spawn_car(start_state)
    
    # Draw Goal
    p.createMultiBody(
        0, -1, 
        p.createVisualShape(p.GEOM_CYLINDER, radius=0.4, length=0.01, rgbaColor=[0, 1, 0, 0.4]), 
        [GOAL_POS[0], GOAL_POS[1], 0.01]
    )

    # Initialize Planner (ensure your collision_checker fix from before is applied!)
    planner = RRTStar(start_state, GOAL_POS, model, env, maze=False)

    print("Planning straight-line RRT*...")
    solve_start = time.time()
    
    # If SMOOTH is False, generate_trajectory usually returns RRT_path for both outputs
    RRT_trajectory, trajectory = generate_trajectory(
        planner, SMOOTHING_FACTOR, TARGET_VELOCITY, MAX_ATTEMPTS, MAX_ITERATIONS
    )

    solve_end = time.time()
    simulation_log['metadata']['computation_time'] = solve_end - solve_start

    if RRT_trajectory is not None:
        print(f"Path found! Computation time: {solve_end - solve_start:.2f}s")
        travel_start = time.time()

        # Log and Draw the Blue RRT path
        for i, node in enumerate(RRT_trajectory):
            simulation_log['planned_rrt_path'].append({
                'x': float(node.x), 'y': float(node.y), 'yaw': float(node.yaw)
            })
            if i < len(RRT_trajectory) - 1:
                p.addUserDebugLine([RRT_trajectory[i].x, RRT_trajectory[i].y, 0.1], 
                                   [RRT_trajectory[i+1].x, RRT_trajectory[i+1].y, 0.1], [0, 0, 1], 2)

        # Execution loop using the RRT nodes directly
        for i, node in enumerate(RRT_trajectory):
            p.resetBasePositionAndOrientation(
                vehicle, [node.x, node.y, 0.15], p.getQuaternionFromEuler([0, 0, node.yaw])
            )

            simulation_log['executed_trajectory'].append({
                'step': i, 'x': float(node.x), 'y': float(node.y),
                'yaw': float(node.yaw), 'v': TARGET_VELOCITY
            })
            
            p.stepSimulation()
            time.sleep(SIM_SLEEP_TIME)

        travel_end = time.time()
        simulation_log['metadata']['travel_time'] = travel_end - travel_start
        save_simulation_log(simulation_log)
        print("Simulation complete.")
    else:
        print("Failed to find path.")

    while True:
        p.stepSimulation()
        time.sleep(0.01)

if __name__ == "__main__":
    main_RRT()