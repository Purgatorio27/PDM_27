
"""
MPC + RRT* Simulation with Dynamic Obstacles (30x30 Map)
Using the integrated MPC_RRT_Controller
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MPC'))

import pybullet as p
import time
import numpy as np
import math
import json
from datetime import datetime

from shared_config import (
    MAP_SIZE, START_POS, GOAL_POS, START_YAW,
    STATIC_OBSTACLES, VEHICLE_RADIUS,
    MAX_SPEED, MIN_SPEED, VEHICLE_LENGTH, VEHICLE_WIDTH
)

# Patch MPC Config
import MPC.Config as mpc_cfg
mpc_cfg.goal = GOAL_POS
mpc_cfg.vehicle_length = VEHICLE_LENGTH
mpc_cfg.vehicle_width = VEHICLE_WIDTH
mpc_cfg.vehicle_radius = VEHICLE_RADIUS
mpc_cfg.lf = VEHICLE_LENGTH * 0.5
mpc_cfg.lr = VEHICLE_LENGTH * 0.5

# Import integrated controller
from MPC.MPC_RRT import create_mpc_rrt_controller
from comparison.run_dynamic_mpc import setup_environment, create_vehicle, create_static_obstacles, get_mpc_static_obstacles
from comparison.dynamic_sim_helper import DynamicObstacleManager

# Parameters
CONTROLLER_DT = 0.05
MAX_STEPS = 1000
HEADLESS = '--headless' in sys.argv

def main():
    print("="*50)
    print("Run Dynamic MPC + RRT* (30x30)")
    print("="*50)
    
    setup_environment(HEADLESS)
    car_id = create_vehicle()
    create_static_obstacles()
    
    # Dynamic Obstacles
    dyn_manager = DynamicObstacleManager(seed=123)
    dyn_visuals = []
    if not HEADLESS:
        for _ in range(dyn_manager.count):
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=dyn_manager.radius, rgbaColor=[1, 0, 0, 1])
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=dyn_manager.radius)
            bid = p.createMultiBody(0, col, vis, [0,0,-10]) 
            dyn_visuals.append(bid)
            
    # MPC+RRT Setup
    static_obs_mpc = get_mpc_static_obstacles()
    dynamic_obs_mpc = dyn_manager.get_for_mpc()
    
    # Create Controller (Plans RRT* internally on init/first solve)
    controller = create_mpc_rrt_controller(
        start_state=[START_POS[0], START_POS[1], 1.0, START_YAW],
        goal=GOAL_POS,
        static_obstacles=static_obs_mpc, # MPC expects dicts
        dynamic_obstacles=dynamic_obs_mpc,
        map_size=MAP_SIZE
    )
    
    car_state = np.array([START_POS[0], START_POS[1], 1.0, START_YAW])
    
    log_data = {
        'metadata': {
                'planner': 'MPC_RRT_Dynamic',
                'map_size': MAP_SIZE,
                'goal': GOAL_POS,
                'start_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            },
        'trajectory': [],
        'controls': [],
        'collisions': 0
    }
    
    collisions = 0
    
    try:
        for step in range(MAX_STEPS):
            # Update Dynamic Obstacles
            dyn_manager.update(CONTROLLER_DT)
            dyn_obs_mpc = dyn_manager.get_for_mpc()
            
            # Sync Visuals
            if not HEADLESS:
                for idx, obs in enumerate(dyn_manager.obstacles):
                    p.resetBasePositionAndOrientation(dyn_visuals[idx], [obs['x'], obs['y'], 0.5], [0,0,0,1])
            
            # --- CRITICAL: Update Controller's Dynamic Obstacles ---
            # Update ALL solvers in the pool just in case logic switches solver
            if hasattr(controller, 'solver_pool'):
                for solver in controller.solver_pool:
                    solver.dynamic_obstacles = dyn_obs_mpc
            elif hasattr(controller, 'mpc'):
                controller.mpc.dynamic_obstacles = dyn_obs_mpc
            
            # Solve
            solve_start = time.time()
            # MPC_RRT solve takes (state, debug...)
            optimal_control, status, debug_info = controller.solve(car_state, debug=False)
            solve_time = time.time() - solve_start
            
            # Apply
            steer, accel = optimal_control
            
            # Kinematics
            x, y, v, psi = car_state
            beta = math.atan((VEHICLE_LENGTH*0.5 / VEHICLE_LENGTH) * math.tan(steer))
            x_new = x + v * math.cos(psi + beta) * CONTROLLER_DT
            y_new = y + v * math.sin(psi + beta) * CONTROLLER_DT
            v_new = v + accel * CONTROLLER_DT
            v_new = np.clip(v_new, MIN_SPEED, MAX_SPEED)
            psi_new = psi + (v / VEHICLE_LENGTH*0.5) * math.sin(beta) * CONTROLLER_DT
            
            car_state = np.array([x_new, y_new, v_new, psi_new])
            
            # Sync Car
            if not HEADLESS:
                p.resetBasePositionAndOrientation(car_id, [x_new, y_new, 0.5], p.getQuaternionFromEuler([0,0,psi_new]))
                time.sleep(1/240.0)
            
            # Log
            log_data['trajectory'].append({
                'step': step, 'x': float(x_new), 'y': float(y_new), 'v': float(v_new), 
                'dynamic_obstacles': dyn_manager.get_for_log()
            })
            log_data['controls'].append({
               'step': step, 'solve_time_ms': solve_time * 1000, 
               'solver_status': int(status)
            })
            
            # Collisions
            for obs in STATIC_OBSTACLES:
               d = math.hypot(x_new - obs['position'][0], y_new - obs['position'][1]) - obs['radius'] - VEHICLE_RADIUS
               if d < 0: collisions += 1
            for dobs in dyn_manager.obstacles:
               d = math.hypot(x_new - dobs['x'], y_new - dobs['y']) - dobs['radius'] - VEHICLE_RADIUS
               if d < 0: collisions += 1

            # Goal
            dist_to_final_goal = math.hypot(x_new - GOAL_POS[0], y_new - GOAL_POS[1])
            if dist_to_final_goal < 1.0:
                print(f"Goal Reached at step {step}!")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        log_data['collisions'] = collisions
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        filename = f"dynamic_mpc_rrt_{log_data['metadata']['start_time']}.json"
        with open(os.path.join(log_dir, filename), 'w') as f:
            json.dump(log_data, f)
        print(f"Log saved to {filename}")
        p.disconnect()

if __name__ == '__main__':
    main()
