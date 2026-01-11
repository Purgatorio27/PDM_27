from Trajectory_generator_obs import generate_obs_trajectory
import numpy as np
import casadi as ca
from Config import max_steps, controller_dt, sim_dt, sim_speed, vehicle_radius

barrel_radius = 1.2  # radius of barrel obstacles
block_radius = 1.4   # radius of block obstacles
human_radius = 1.0   # radius of humans (dynamic obstacles)

# Static obstacle parameters  
static_obstacles = []
static_obstacles.append({'position': [8, 8], 'type': 'circle', 'radius': barrel_radius})
# static_obstacles.append({'position': [7, 24], 'type': 'square', 'radius': block_radius})
# static_obstacles.append({'position': [17, 5], 'type': 'square', 'radius': block_radius})
static_obstacles.append({'position': [26, 26], 'type': 'circle', 'radius': barrel_radius})

# Random static obstacles 
def add_random_static(static_obstacles, max_obstacles = 18):
    """Generates workshop clusters while ensuring no overlaps."""        
    safe_zones = [np.array([0.0, 0.0]), np.array([40.0, 40.0])]  # No obstacles on start and goal
    
    clusters_placed = 0
    attempts = 0

    while clusters_placed < max_obstacles and attempts < 1200:
        attempts += 1
        cx = np.random.uniform(4.0, 36.0)
        cy = np.random.uniform(4.0, 36.0)
        new_pos = np.array([cx, cy])

        # Check distance against start/goal
        too_close = False
        for zone in safe_zones:
            if np.linalg.norm(new_pos - zone) < 4.0: 
                too_close = True
                break
        if too_close: continue
        
        # Check distance against existing obstacles to prevent overlap
        for obs in static_obstacles:
            obs_pos = np.array(obs['position'])
    
            if np.linalg.norm(new_pos - obs_pos) < 2 * (vehicle_radius + barrel_radius):
                too_close = True
                break
        
        if not too_close:
        
            template = np.random.choice(['barrel', 'block'])
            
            if template == 'barrel':
                static_obstacles.append({
                    'position': [cx, cy], 
                    'type': 'circle', 
                    'radius': barrel_radius
                })
            
            elif template == 'block':
                static_obstacles.append({
                    'position': [cx, cy], 
                    'type': 'square', 
                    'radius': block_radius
                })
            
            clusters_placed += 1

    return static_obstacles

static_obstacles = add_random_static(static_obstacles, max_obstacles = 7)



# Dynamic obstacle parameters  -- add your dynamic obstacles here
dynamic_obstacles = []

safety_margin = 0.3 

total_sim_time = controller_dt * max_steps
obs_traj_steps = int(total_sim_time / sim_dt) + 1

def is_traj_collision_free(candidate_traj, static_obs, existing_dyn_obs, radius):

    """collsion check for candidate dynamic obstacle trajectory"""
    for t in range(len(candidate_traj)):
        pos_t = candidate_traj[t, :2]
        
        # 1. with static obstacles
        for s_obs in static_obs:
            s_pos = np.array(s_obs['position'])
            if np.linalg.norm(pos_t - s_pos) < (radius + s_obs['radius'] + safety_margin):
                return False
        
        # 2. with existing dynamic obstacles (only check up to current time t)
        for d_obs in existing_dyn_obs:
            d_traj = d_obs['trajectory']
            
            if t < len(d_traj):
                d_pos_t = d_traj[t, :2]
                if np.linalg.norm(pos_t - d_pos_t) < (radius + d_obs['radius'] + safety_margin):
                    return False
                    
        # 3. boundary check
        if not (0.6 <= pos_t[0] <= 39.5 and 0.6 <= pos_t[1] <= 39.5):
            return False
            
    return True

num_dynamic_needed = 8  # dynamic obstacles
attempts = 0
max_total_attempts = 1500

while len(dynamic_obstacles) < num_dynamic_needed and attempts < max_total_attempts:
    attempts += 1
    
    # random start position away from start/goal
    start_pos = [np.random.uniform(2.0, 38.0), np.random.uniform(2.0, 38.0)]
    if np.linalg.norm(np.array(start_pos) - np.array([0,0])) < 5.0 or \
       np.linalg.norm(np.array(start_pos) - np.array([40,40])) < 5.0:
        continue

    # random reference velocity
    v_mag = np.random.uniform(0.1, 0.6) * sim_speed
    v_angle = np.random.uniform(0, 2 * np.pi)
    velocity_ref = [v_mag * np.cos(v_angle), v_mag * np.sin(v_angle)]

    # generate candidate trajectory
    candidate_traj = generate_obs_trajectory(start_pos, velocity_ref, num_steps=obs_traj_steps, dt=sim_dt)

    # validate trajectory for collision-free 
    if is_traj_collision_free(candidate_traj, static_obstacles, dynamic_obstacles, human_radius):
        dynamic_obstacles.append({
            'reference velocity': velocity_ref,
            'trajectory': candidate_traj,
            'radius': human_radius
        })

print(f"Successfully generated {len(dynamic_obstacles)} dynamic obstacles.")



def distance(pos1, pos2, radius1=0.0, radius2=0.0):
    """
    Compute obstacle distance (considering radius) - CasADi version  
    """
    dist = ca.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) - (radius1 + radius2)
    return dist