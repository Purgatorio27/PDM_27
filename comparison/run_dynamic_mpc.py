
"""
Pure MPC Simulation with Dynamic Obstacles (30x30 Map)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MPC'))

import pybullet as p
import pybullet_data
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

# Patch MPC Config before importing MPC_core
import MPC.Config as mpc_cfg
mpc_cfg.goal = GOAL_POS
mpc_cfg.vehicle_length = VEHICLE_LENGTH
mpc_cfg.vehicle_width = VEHICLE_WIDTH
mpc_cfg.vehicle_radius = VEHICLE_RADIUS
mpc_cfg.lf = VEHICLE_LENGTH * 0.5
mpc_cfg.lr = VEHICLE_LENGTH * 0.5

from MPC.MPC_core import MPC
from comparison.dynamic_sim_helper import DynamicObstacleManager

# Parameters
CONTROLLER_DT = 0.05
MAX_STEPS = 1000
MPC_HORIZON = 40
MPC_DT = 0.2
HEADLESS = '--headless' in sys.argv

def setup_environment(headless=False):
    if headless:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI)
    
    p.setGravity(0, 0, -9.8)
    p.resetDebugVisualizerCamera(cameraDistance=MAP_SIZE, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[MAP_SIZE/2, MAP_SIZE/2, 0])
    
    # Ground
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
    
    # Walls
    wall_height = 2.0
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[MAP_SIZE/2, 0.1, wall_height/2])
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[MAP_SIZE/2, 0.1, wall_height/2], rgbaColor=[0.5, 0.5, 0.5, 1])
    
    p.createMultiBody(0, col_id, vis_id, [MAP_SIZE/2, 0, wall_height/2]) # Bottom
    p.createMultiBody(0, col_id, vis_id, [MAP_SIZE/2, MAP_SIZE, wall_height/2]) # Top
    
    col_id_v = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, MAP_SIZE/2, wall_height/2])
    vis_id_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, MAP_SIZE/2, wall_height/2], rgbaColor=[0.5, 0.5, 0.5, 1])
    
    p.createMultiBody(0, col_id_v, vis_id_v, [0, MAP_SIZE/2, wall_height/2]) # Left
    p.createMultiBody(0, col_id_v, vis_id_v, [MAP_SIZE, MAP_SIZE/2, wall_height/2]) # Right

def create_vehicle():
    car_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[VEHICLE_LENGTH/2, VEHICLE_WIDTH/2, 0.5], rgbaColor=[0, 0, 1, 1])
    car_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[VEHICLE_LENGTH/2, VEHICLE_WIDTH/2, 0.5])
    start_orientation = p.getQuaternionFromEuler([0, 0, START_YAW])
    car_id = p.createMultiBody(baseMass=1000, baseCollisionShapeIndex=car_col, baseVisualShapeIndex=car_visual, 
                               basePosition=[START_POS[0], START_POS[1], 0.5], baseOrientation=start_orientation)
    return car_id

def create_static_obstacles():
    obs_ids = []
    for obs in STATIC_OBSTACLES:
        radius = obs['radius']
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=2.0, rgbaColor=[0.4, 0.4, 0.4, 1])
        col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=2.0)
        pos = [obs['position'][0], obs['position'][1], 1.0]
        obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=visual_shape, basePosition=pos)
        obs_ids.append(obs_id)
    return obs_ids

def get_mpc_static_obstacles():
    obstacles = []
    for obs in STATIC_OBSTACLES:
        obstacles.append({
            'position': obs['position'],
            'type': 'circle',
            'radius': obs['radius']
        })
    return obstacles

def compute_min_obstacle_distance(car_pos, static_obstacles, dynamic_obstacles):
    min_dist = float('inf')
    # Static
    for obs in static_obstacles:
        ox, oy = obs['position']
        dist = math.hypot(car_pos[0] - ox, car_pos[1] - oy) - obs['radius'] - VEHICLE_RADIUS
        min_dist = min(min_dist, dist)
    
    # Dynamic
    for obs in dynamic_obstacles:
        # Dynamic obs structure from DynamicObstacleManager: {'x', 'y', ...}
        # BUT here we might receive the MPC formatted list if we access controller directly
        # Let's use the explicit check in main loop
        pass
        
    return min_dist

def main():
    print("="*50)
    print("Run Dynamic MPC (30x30)")
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
            bid = p.createMultiBody(0, col, vis, [0,0,-10]) # Init off-screen
            dyn_visuals.append(bid)
            
    # MPC Setup
    static_obs_mpc = get_mpc_static_obstacles()
    # Init with first frame of dynamic
    dynamic_obs_mpc = dyn_manager.get_for_mpc()
    
    mpc_controller = MPC(static_obs_mpc, dynamic_obs_mpc, horizon=MPC_HORIZON, dt=MPC_DT, goal=GOAL_POS)
    
    car_state = np.array([START_POS[0], START_POS[1], 1.0, START_YAW])
    
    # Logging
    log_data = {
        'metadata': {
                'planner': 'MPC_Dynamic',
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
            
            # Update PyBullet Visuals
            if not HEADLESS:
                for idx, obs in enumerate(dyn_manager.obstacles):
                    p.resetBasePositionAndOrientation(dyn_visuals[idx], [obs['x'], obs['y'], 0.5], [0,0,0,1])
                    
            # Pass Updated Dynamic Obstacles to MPC
            mpc_controller.dynamic_obstacles = dyn_obs_mpc
            
            # Solve MPC
            solve_start = time.time()
            optimal_control, status, _ = mpc_controller.solve(car_state, debug=False)
            solve_time = time.time() - solve_start
            
            # Apply Control (Simulation)
            steer, accel = optimal_control
            
            # Update Car State (Kinematic Model)
            # x, y, v, psi
            x, y, v, psi = car_state
            
            # Simple bicycle model integration
            beta = math.atan((VEHICLE_LENGTH*0.5 / VEHICLE_LENGTH) * math.tan(steer))
            x_new = x + v * math.cos(psi + beta) * CONTROLLER_DT
            y_new = y + v * math.sin(psi + beta) * CONTROLLER_DT
            v_new = v + accel * CONTROLLER_DT
            v_new = np.clip(v_new, MIN_SPEED, MAX_SPEED)
            psi_new = psi + (v / VEHICLE_LENGTH*0.5) * math.sin(beta) * CONTROLLER_DT
            
            car_state = np.array([x_new, y_new, v_new, psi_new])
            
            # Sync PyBullet Car
            if not HEADLESS:
                p.resetBasePositionAndOrientation(car_id, [x_new, y_new, 0.5], p.getQuaternionFromEuler([0,0,psi_new]))
                time.sleep(1/240.0) # Speed up 
            
            # Log
            log_data['trajectory'].append({
                'step': step, 'x': float(x_new), 'y': float(y_new), 'v': float(v_new), 
                'dynamic_obstacles': dyn_manager.get_for_log()
            })
            log_data['controls'].append({
               'step': step, 'solve_time_ms': solve_time * 1000, 
               'solver_status': int(status)
            })

            # Check Collisions
            min_dist = float('inf')
            # Check static
            for obs in STATIC_OBSTACLES:
               d = math.hypot(x_new - obs['position'][0], y_new - obs['position'][1]) - obs['radius'] - VEHICLE_RADIUS
               if d < 0: collisions += 1
               min_dist = min(min_dist, d)
               
            # Check dynamic
            for dobs in dyn_manager.obstacles:
               d = math.hypot(x_new - dobs['x'], y_new - dobs['y']) - dobs['radius'] - VEHICLE_RADIUS
               if d < 0: collisions += 1
               min_dist = min(min_dist, d)
            
            # Check Goal
            dist_to_goal = math.hypot(x_new - GOAL_POS[0], y_new - GOAL_POS[1])
            if dist_to_goal < 1.0:
                print(f"Goal Reached at step {step}!")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        log_data['collisions'] = collisions
        # Save Log
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        filename = f"dynamic_mpc_{log_data['metadata']['start_time']}.json"
        with open(os.path.join(log_dir, filename), 'w') as f:
            json.dump(log_data, f)
        print(f"Log saved to {filename}")
        p.disconnect()

if __name__ == '__main__':
    main()
