import sys
import os
import pybullet as p
import time
import math
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from KinematicBicycleModelRRT import State
from RRTStar import RRTStar 

try:
    from Environment.environment import MazeEnvironment
except ModuleNotFoundError:
    sys.path.append(os.path.join(root_dir, "Environment"))
    from environment import MazeEnvironment

# Simple holonomic model
class SimplePointModel:
    def next_state(self, state, target_pos, step_size):
        dx = target_pos[0] - state.x
        dy = target_pos[1] - state.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6: return state
        new_yaw = math.atan2(dy, dx)
        new_x = state.x + step_size * (dx / dist)
        new_y = state.y + step_size * (dy / dist)
        return State(new_x, new_y, new_yaw)

# Parameters
DIFFICULTY = "Master"  # Choose: "Simple", "Intermediate", "Advanced" ("Expert" and "Master" exist but are too difficult)
MAX_ITERATIONS = 20000
MAX_ATTEMPTS = 3
TARGET_VELOCITY = 5.0
SIM_HERTZ = 240

def main_no_steer():
    env = MazeEnvironment(difficulty=DIFFICULTY)
    model = SimplePointModel()
    
    START_POSE = env.config['start']
    GOAL_POS = env.config['goal']
    start_state = State(START_POSE[0], START_POSE[1], START_POSE[2])
    
    vehicle = env.spawn_car(START_POSE)
    env.draw_goal(GOAL_POS)

    planner = RRTStar(start_state, GOAL_POS, model, env)
    planner.search_radius = 2.0 
    
    print(f"Planning")
    path = None
    for attempt in range(MAX_ATTEMPTS):
        path = planner.plan(max_iter=MAX_ITERATIONS)
        if path: break

    if path:
        print(f"Path found. Visualizing and executing")
        for i in range(len(path) - 1):
            p.addUserDebugLine([path[i].x, path[i].y, 0.1], [path[i+1].x, path[i+1].y, 0.1], [0, 1, 0], 3)

        # Instead of jumping from node to node, interpolate
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i+1]
            
            # Calculate distance between these two nodes
            segment_dist = math.hypot(end_node.x - start_node.x, end_node.y - start_node.y)
            
            # Calculate how many simulation steps this segment should take: steps = distance / velocity * frequency
            num_steps = max(1, int((segment_dist / TARGET_VELOCITY) * SIM_HERTZ))

            for s in range(num_steps):
                # Interpolation factor (0 to 1)
                t = s / num_steps
                
                # Linearly interpolate position
                curr_x = start_node.x + (end_node.x - start_node.x) * t
                curr_y = start_node.y + (end_node.y - start_node.y) * t
                
                # Use the heading of the segment
                curr_yaw = end_node.yaw 

                p.resetBasePositionAndOrientation(
                    vehicle, 
                    [curr_x, curr_y, 0.15], 
                    p.getQuaternionFromEuler([0, 0, curr_yaw])
                )
                
                p.stepSimulation()
                time.sleep(1/SIM_HERTZ)  # 1/240

        print("Goal reached")
    else:
        print("No path found")

    while p.isConnected():
        p.stepSimulation()
        time.sleep(0.01)

if __name__ == "__main__":
    main_no_steer()