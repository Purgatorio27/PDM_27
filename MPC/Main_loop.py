import pybullet as p
import pybullet_data
import time
import numpy as np
import math

# Import existing modules
from Obstacles import static_obstacles, dynamic_obstacles
from Config import T, max_steps, dt, goal, lf, lr, vehicle_length, vehicle_width, vehicle_radius, \
    max_speed, min_speed, max_acceleration, max_deceleration, max_steering_angle, \
    controller_dt, sim_dt, sim_speed
from MPC_core import MPC

# Initialize PyBullet simulation
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(dt)

# Load plane and vehicle model
planeId = p.loadURDF("plane.urdf", [0, 0, 0])

p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[8,10,0])
car_half_extents = [vehicle_length / 2, vehicle_width / 2, 0.05]
collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=vehicle_radius)  # Spherical collision shape
car_visualized = p.createVisualShape(p.GEOM_BOX, halfExtents=car_half_extents, rgbaColor=[1, 0, 0, 1])  

car_id = p.createMultiBody(
    baseMass = 1.0,
    baseCollisionShapeIndex = collision_shape,
    baseVisualShapeIndex = car_visualized,\
    basePosition=[0, 0, 0.1]  # caution: z should be > 0 to avoid initial collision with ground
)


# Initialize Static Obstacles in PyBullet
stat_body_ids = []

for obs in static_obstacles:
    pos_3d = list(obs['position']) + [0.1] # z = 0.1

    # Spherical equivalent for both circle and square obstacles
    col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=obs['radius'], height=0.2)
    
    # Visual shape differs based on type
    if obs['type'] == 'circle':
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=obs['radius'], length = 0.2, rgbaColor=[0, 0, 0, 1.0])  # green circle
    else:  # square 
        half_side = obs['radius'] / math.sqrt(2)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_side, half_side, 0.05], rgbaColor=[0, 0, 1, 0.8])  # blue square

    body_id = p.createMultiBody(
        baseMass = 0.0,  # static
        baseCollisionShapeIndex = col_shape,
        baseVisualShapeIndex = visual_shape,
        basePosition = pos_3d
    )
    stat_body_ids.append(body_id)


# Initialize dynamic obstacles in PyBullet
dyn_body_ids = []

for d_obs in dynamic_obstacles:

    initial_pos = d_obs['trajectory'][0][0:2].tolist() + [0.1]
    
    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=d_obs['radius'])
    visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=d_obs['radius'], length = 0.2, rgbaColor=[1, 1, 0, 0.8])  
    
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=initial_pos
    )
    dyn_body_ids.append(body_id)

all_obstacle_ids = stat_body_ids + dyn_body_ids


# Simulation engine (state update)
def vehicle_simulate_engine(state, control, dt):
    x, y, v, psi = state
    delta, a = control

    # Kinematic bicycle model update
    x_new = x + v * math.cos(psi) * dt
    y_new = y + v * math.sin(psi) * dt

    v_new = v + a * dt
    # Apply speed limits (EXTRA SAFE CHECK)
    v_new = np.clip(v_new, min_speed, max_speed)

    beta = math.atan((lr / (lf + lr)) * math.tan(delta))
    psi_new = psi + (v / lr) * math.sin(beta) * dt 

    return np.array([x_new, y_new, v_new, psi_new])


def compute_emergency_avoidance(car_state, static_obs, dyn_obs, goal_pos):
    """
    Compute emergency avoidance control when deadlock is detected.
    Steers aggressively away from the nearest obstacle while maintaining speed.
    """
    x, y, v, psi = car_state
    
    # Find the nearest obstacle
    min_dist = float('inf')
    nearest_obs_pos = None
    
    for obs in static_obs:
        ox, oy = obs['position']
        dist = math.hypot(x - ox, y - oy) - obs['radius'] - vehicle_radius
        if dist < min_dist:
            min_dist = dist
            nearest_obs_pos = (ox, oy)
    
    for obs in dyn_obs:
        ox, oy = obs['trajectory'][0], obs['trajectory'][1]
        dist = math.hypot(x - ox, y - oy) - obs['radius'] - vehicle_radius
        if dist < min_dist:
            min_dist = dist
            nearest_obs_pos = (ox, oy)
    
    if nearest_obs_pos is None:
        # No obstacles, steer towards goal
        goal_angle = math.atan2(goal_pos[1] - y, goal_pos[0] - x)
        heading_error = goal_angle - psi
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        steer = np.clip(heading_error * 1.5, -max_steering_angle, max_steering_angle)
        return np.array([steer, 1.5])
    
    ox, oy = nearest_obs_pos
    
    # Vector from obstacle to car
    dx = x - ox
    dy = y - oy
    angle_from_obs = math.atan2(dy, dx)
    
    # Choose perpendicular direction that moves towards goal
    goal_angle = math.atan2(goal_pos[1] - y, goal_pos[0] - x)
    
    # Two perpendicular escape directions
    perp_angle1 = angle_from_obs + math.pi / 2
    perp_angle2 = angle_from_obs - math.pi / 2
    
    # Pick the one closer to goal
    diff1 = abs(math.atan2(math.sin(goal_angle - perp_angle1), math.cos(goal_angle - perp_angle1)))
    diff2 = abs(math.atan2(math.sin(goal_angle - perp_angle2), math.cos(goal_angle - perp_angle2)))
    
    target_angle = perp_angle1 if diff1 < diff2 else perp_angle2
    
    # Compute aggressive steering
    heading_error = target_angle - psi
    heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
    
    # Very aggressive steering during recovery
    steer = np.clip(heading_error * 2.0, -max_steering_angle, max_steering_angle)
    
    # Maintain positive acceleration to get out of deadlock
    if min_dist < 0.5:
        accel = -0.5  # slight brake if very close
    else:
        accel = 1.5  # accelerate to escape
    
    return np.array([steer, accel])


# Initialize dynamic obstacle state at step 0 (to feed MPC_core)
cur_dyn_obs_MPC = []
for d_obs in dynamic_obstacles:
    cur_dyn_obs_MPC.append({
        'trajectory': d_obs['trajectory'][0],
        'radius': d_obs['radius']
    })

mpc_controller = MPC(static_obstacles, cur_dyn_obs_MPC, horizon = 20, dt = dt)

# Final simulation setup
goal_tolerance = 0.3

# Debug visualization setup
goal_marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 0, 0.8])
goal_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_marker, basePosition=[goal[0], goal[1], 0.1])

# Debug text ID for efficient updates
debug_text_id = None

def update_debug_params(car_state, control, dist_to_goal, min_obs_dist, solver_status, step):
    """Update debug text overlay (more efficient than recreating parameters)"""
    global debug_text_id

    x, y, v, psi = car_state
    delta, a = control

    # Remove old text
    if debug_text_id is not None:
        p.removeUserDebugItem(debug_text_id)

    # Create single debug text block
    text = (f"Step:{step} | Pos:({x:.1f},{y:.1f}) | V:{v:.2f}m/s | Hdg:{np.degrees(psi):.0f}deg\n"
            f"Steer:{np.degrees(delta):.1f}deg | Accel:{a:.2f} | Goal:{dist_to_goal:.1f}m | Obs:{min_obs_dist:.1f}m | Solver:{solver_status}")

    debug_text_id = p.addUserDebugText(
        text, [0, 0, 3], textColorRGB=[1, 1, 0], textSize=1.5
    )


def compute_min_obstacle_distance(car_pos, static_obs, dyn_obs_positions):
    """Compute minimum distance to any obstacle"""
    min_dist = float('inf')

    for obs in static_obs:
        ox, oy = obs['position']
        dist = math.hypot(car_pos[0] - ox, car_pos[1] - oy) - obs['radius'] - vehicle_radius
        min_dist = min(min_dist, dist)

    for i, d_obs in enumerate(dynamic_obstacles):
        ox, oy = dyn_obs_positions[i][0], dyn_obs_positions[i][1]
        dist = math.hypot(car_pos[0] - ox, car_pos[1] - oy) - d_obs['radius'] - vehicle_radius
        min_dist = min(min_dist, dist)

    return min_dist


# ======= Main Loop =======
def __main__():

    # Initialize car state
    car_state = np.array([0.0, 0.0, 1.0, np.radians(45)])  # x, y, v, psi
    solver_status = 0
    
    # Deadlock detection variables
    low_speed_counter = 0
    LOW_SPEED_THRESHOLD = 0.3  # m/s
    DEADLOCK_STEPS = 20  # consecutive low-speed steps to trigger recovery
    last_position = np.array([0.0, 0.0])
    position_change_threshold = 0.1  # meters
    deadlock_recovery_mode = False
    recovery_steps = 0
    RECOVERY_DURATION = 30  # steps to stay in recovery mode

    for step in range(max_steps):
        # Compute current simulation time and obstacle trajectory index
        current_time = step * controller_dt
        obs_traj_idx = int(current_time / sim_dt)
        obs_traj_idx = min(obs_traj_idx, len(dynamic_obstacles[0]['trajectory']) - 1)  # clamp to valid range

        # Update dynamic obstacles positions in PyBullet
        for i, dyn_id in enumerate(dyn_body_ids):
            current_pos_2d = dynamic_obstacles[i]['trajectory'][obs_traj_idx][0:2]
            pos_3d = current_pos_2d.tolist() + [0.1]
            p.resetBasePositionAndOrientation(dyn_id, pos_3d, [0, 0, 0, 1])

        cur_state_np = np.array(car_state)

        cur_dyn_obs_MPC = []
        for d_obs in dynamic_obstacles:
            cur_dyn_obs_MPC.append({
                'trajectory': d_obs['trajectory'][obs_traj_idx],
                'radius': d_obs['radius']
            })

        mpc_controller.dynamic_obstacles = cur_dyn_obs_MPC

        #Solve MPC
        optimal_control, solver_status = mpc_controller.solve(cur_state_np)
        
        # ===== Deadlock Detection and Recovery =====
        current_speed = abs(car_state[2])
        position_change = np.linalg.norm(car_state[0:2] - last_position)
        
        # Check if vehicle is stuck (low speed and not moving much)
        if current_speed < LOW_SPEED_THRESHOLD and position_change < position_change_threshold:
            low_speed_counter += 1
        else:
            low_speed_counter = 0
            if not deadlock_recovery_mode:
                last_position = car_state[0:2].copy()
        
        # Trigger recovery mode if deadlock detected
        if low_speed_counter >= DEADLOCK_STEPS and not deadlock_recovery_mode:
            deadlock_recovery_mode = True
            recovery_steps = 0
            print(f"[Step {step}] *** DEADLOCK DETECTED! Entering recovery mode ***")
        
        # Apply emergency avoidance control during recovery
        if deadlock_recovery_mode:
            recovery_steps += 1
            
            # Compute emergency steering away from nearest obstacle
            emergency_control = compute_emergency_avoidance(
                car_state, static_obstacles, cur_dyn_obs_MPC, goal
            )
            optimal_control = emergency_control
            
            # Exit recovery mode after enough steps or if moving again
            if recovery_steps >= RECOVERY_DURATION or (current_speed > 1.0 and position_change > 0.5):
                deadlock_recovery_mode = False
                low_speed_counter = 0
                print(f"[Step {step}] Recovery mode ended, resuming normal control")
        
        last_position = car_state[0:2].copy()

        # Update vehicle state
        car_state = vehicle_simulate_engine(car_state, optimal_control, dt)

        # Update vehicle position and orientation in PyBullet
        car_pos = [car_state[0], car_state[1], 0.1]
        car_orn = p.getQuaternionFromEuler([0, 0, car_state[3]])
        p.resetBasePositionAndOrientation(car_id, car_pos, car_orn)

        # Compute distances for debug
        dist_to_goal = math.hypot(car_state[0] - goal[0], car_state[1] - goal[1])

        # Get current dynamic obstacle positions for distance calculation
        dyn_obs_positions = []
        for d_obs in dynamic_obstacles:
            dyn_obs_positions.append(d_obs['trajectory'][obs_traj_idx][0:2])
        min_obs_dist = compute_min_obstacle_distance(car_state[0:2], static_obstacles, dyn_obs_positions)

        # Update debug display (only every 5 steps for performance)
        if step % 5 == 0:
            update_debug_params(car_state, optimal_control, dist_to_goal, min_obs_dist, solver_status, step)

        # Collision detection
        collided_indicator = False

        for obs_id in all_obstacle_ids:
            contact_points = p.getContactPoints(car_id, obs_id)
            if len(contact_points) > 0:
                collided_indicator = True
                break

        # if collided_indicator is True:
        #     print(f"*** At {step} step, (t={step*dt:.1f}s) Collision occurred! ***")
        #     break

        # Check goal reached
        if dist_to_goal < goal_tolerance:
            print(f"*** At {step} step, (t={step*dt:.1f}s) Goal reached! ***")
            time.sleep(2)
            break

        # pybullet setup (extra)
        p.stepSimulation()
        time.sleep(0.01)  # small delay for visualization, increase if too fast
        
    # End of simulation loop
    print("Simulation ended.")
    p.disconnect()

if __name__ == "__main__":
    __main__()
    






