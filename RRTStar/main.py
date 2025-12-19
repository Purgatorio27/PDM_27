import pybullet as p
import time
import numpy as np

from KinematicBicycleModelRRT import KinematicBicycleModelRRT, State
from Environment import Environment
from RRTStar import RRTStar
from utils import generate_trajectory


# All parameters
USE_RANDOM_ENV = False  # Probably keep it at false if we use dynamic obstacles
MIN_TURN_RAD = 0.8

START_POSE = (-4.2, -4.2, 0.78)  # x, y, yaw
GOAL_POS = (4.2, 4.2)

MAX_ITERATIONS = 15000
MAX_ATTEMPTS = 5

SMOOTH = False
SMOOTHING_FACTOR = 30

TARGET_VELOCITY = 10.0

SIM_SLEEP_TIME = 0.015


def main_RRT():
    env = Environment(randomize=USE_RANDOM_ENV)
    model = KinematicBicycleModelRRT(min_radius=MIN_TURN_RAD)
    
    start_state = State(START_POSE[0], START_POSE[1], START_POSE[2])

    vehicle = env.spawn_car(start_state)
    p.createMultiBody(
        0, -1, 
        p.createVisualShape(p.GEOM_CYLINDER, radius=0.4, length=0.01, rgbaColor=[0, 1, 0, 0.4]), 
        [GOAL_POS[0], GOAL_POS[1], 0.01]
    )

    planner = RRTStar(start_state, GOAL_POS, model, env)
    RRT_trajectory, trajectory = generate_trajectory(
        planner, SMOOTHING_FACTOR, TARGET_VELOCITY, MAX_ATTEMPTS, MAX_ITERATIONS
    )

    print("shape: ", np.shape(trajectory))

    if trajectory is not None:
        for i in range(len(RRT_trajectory) - 1):  # RRT path, always show
            p.addUserDebugLine([RRT_trajectory[i].x, RRT_trajectory[i].y, 0.1], 
                               [RRT_trajectory[i+1].x, RRT_trajectory[i+1].y, 0.1], [0, 0, 1], 1)

        if SMOOTH:  # Smooth path
            for i in range(len(trajectory) - 1):
                p.addUserDebugLine([trajectory[i, 0], trajectory[i, 1], 0.12], 
                                   [trajectory[i+1, 0], trajectory[i+1, 1], 0.12], [1, 0, 0], 3)


        # Make model move along trajectory (teleports every SIM_SLEEP_TIME seconds)
        for point in trajectory:
            target_x, target_y, target_yaw, target_v = point
            
            p.resetBasePositionAndOrientation(
                vehicle, [target_x, target_y, 0.15], p.getQuaternionFromEuler([0, 0, target_yaw])
            )
            
            p.stepSimulation()
            time.sleep(SIM_SLEEP_TIME)
    else:
        print(f"RRT* could not find a viable path within {MAX_ITERATIONS} iterations")
        return

    # Dont close PyBullet at the end of simulation
    while True:
        p.stepSimulation()
        time.sleep(0.01)


if __name__ == "__main__":
    main_RRT()