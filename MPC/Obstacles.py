from Trajectory_generator_obs import generate_obs_trajectory
import numpy as np
import casadi as ca
from Config import max_steps, controller_dt, sim_dt, sim_speed


# Static obstacle parameters  -- add your static obstacles here
static_obstacles = []
static_obstacles.append({'id': 1, 'position': (8, 8), 'type': 'circle', 'radius': 2.0})
static_obstacles.append({'id': 2,'position': (1, 14), 'type': 'square', 'radius': 1.6})
static_obstacles.append({'id': 3, 'position': (17, 5), 'type': 'square', 'radius': 2.6})
static_obstacles.append({'id': 4, 'position': (16, 16), 'type': 'circle', 'radius': 1.5})

# Dynamic obstacle parameters  -- add your dynamic obstacles here
dynamic_obstacles = []
set_radius_1 = 1.0   # radius 1 for dynamic obstacles, cylindrical shape assumed (pedestrians)
set_radius_2 = 0.7   # radius 0.5 for dynamic obstacles, cylindrical shape assumed (pedestrians)

# Calculate total trajectory steps needed (fine-grained for smooth obstacle motion)
total_sim_time = controller_dt * max_steps
obs_traj_steps = int(total_sim_time / sim_dt) + 1

start_pos1 = [5, 5]
velocity_ref1 = [-0.3 * sim_speed, -0.3 * sim_speed]  # scaled by sim_speed
traj1 = generate_obs_trajectory(start_pos1, velocity_ref1, num_steps=obs_traj_steps, dt=sim_dt)

dynamic_obstacles.append({'id': 1, 'reference velocity': velocity_ref1,\
                           'trajectory': traj1, 'radius': set_radius_1})

start_pos2 = [10, 12]
velocity_ref2 = [-0.1 * sim_speed, -0.04 * sim_speed]  # scaled by sim_speed
traj2 = generate_obs_trajectory(start_pos2, velocity_ref2, num_steps=obs_traj_steps, dt=sim_dt)
dynamic_obstacles.append({'id': 2, 'reference velocity': velocity_ref2,\
                           'trajectory': traj2, 'radius': set_radius_1})

start_pos3 = [16, 19]
velocity_ref3 = [0.01 * sim_speed, -0.2 * sim_speed]  # scaled by sim_speed
traj3 = generate_obs_trajectory(start_pos3, velocity_ref3, num_steps=obs_traj_steps, dt=sim_dt)
dynamic_obstacles.append({'id': 3, 'reference velocity': velocity_ref3,\
                           'trajectory': traj3, 'radius': set_radius_2})



def distance(pos1, pos2, radius1=0.0, radius2=0.0):
    """
    Compute obstacle distance (considering radius) - CasADi version  
    """
    dist = ca.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) - (radius1 + radius2)
    return dist