
import sys
import os
import math
sys.path.append(os.path.join(os.getcwd(), 'MPC'))
sys.path.append(os.path.join(os.getcwd(), 'RRTStar'))

from MPC.MPC_RRT import RRTPlanner
import MPC.Config as cfg
import numpy as np

# Set dummy config values if needed (Config might have defaults)
cfg.vehicle_radius = 0.9

def debug_rrt():
    np.random.seed(123)
    start_pos = (2.0, 2.0)
    goal_pos = (26.0, 26.0)
    map_size = 30.0
    
    # Obstacles from shared_config
    static_obstacles = [
        {'position': [8.0, 8.0], 'radius': 1.2},
        {'position': [12.0, 12.0], 'radius': 1.5},
        {'position': [17.0, 17.0], 'radius': 1.2},
        {'position': [5.0, 12.0], 'radius': 1.0},
        {'position': [4.0, 18.0], 'radius': 1.3},
        {'position': [18.0, 6.0], 'radius': 1.0},
    ]
    
    print("Initializing RRTPlanner...")
    planner = RRTPlanner(
        start_pos=start_pos,
        goal_pos=goal_pos,
        static_obstacles=static_obstacles,
        map_size=map_size,
        vehicle_radius=cfg.vehicle_radius,
        safety_margin=0.5
    )
    
    print("Planning...")
    path = planner.plan(max_iter=5000)
    
    if path:
        print(f"Success! Path length: {len(path)}")
        for i, node in enumerate(path):
            print(f"Node {i}: ({node.x:.2f}, {node.y:.2f})")
    else:
        print("Failure!")

if __name__ == "__main__":
    debug_rrt()
