import pybullet as p
import pybullet_data
import time
import numpy as np
import math

# Import existing modules
from Obstacles import static_obstacles, dynamic_obstacles
from Config import T, max_steps, dt, goal, lf, lr, vehicle_length, vehicle_width, vehicle_radius, \
    max_speed, min_speed, max_acceleration, max_deceleration, max_steering_angle
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
    beta = math.atan((lr / (lf + lr)) * math.tan(delta))

    # Kinematic bicycle model update
    x_new = x + v * math.cos(psi + beta) * dt
    y_new = y + v * math.sin(psi + beta) * dt

    v_new = v + a * dt
    # Apply speed limits (EXTRA SAFE CHECK)
    v_new = np.clip(v_new, min_speed, max_speed)

    psi_new = psi + (v / lr) * math.sin(beta) * dt 

    return np.array([x_new, y_new, v_new, psi_new])


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

# ======= Main Loop =======
def __main__():

    # Initialize car state
    car_state = np.array([0.0, 0.0, 1.0, np.radians(45)])  # x, y, v, psi

    for step in range(max_steps):
        # Update dynamic obstacles positions in PyBullet
        for i, dyn_id in enumerate(dyn_body_ids):
            current_pos_2d = dynamic_obstacles[i]['trajectory'][step][0:2]
            pos_3d = current_pos_2d.tolist() + [0.1]
            p.resetBasePositionAndOrientation(dyn_id, pos_3d, [0, 0, 0, 1])
    
        cur_state_np = np.array(car_state)

        cur_dyn_obs_MPC = []
        for d_obs in dynamic_obstacles:
            cur_dyn_obs_MPC.append({
                'trajectory': d_obs['trajectory'][step],
                'radius': d_obs['radius']
            })

        mpc_controller.dynamic_obstacles = cur_dyn_obs_MPC

        #Solve MPC
        optimal_control = mpc_controller.solve(cur_state_np)

        # Update vehicle state
        car_state = vehicle_simulate_engine(car_state, optimal_control, dt)

        # Update vehicle position and orientation in PyBullet
        car_pos = [car_state[0], car_state[1], 0.1]
        car_orn = p.getQuaternionFromEuler([0, 0, car_state[3]])
        p.resetBasePositionAndOrientation(car_id, car_pos, car_orn)
        
        # Collision detection
        collided_indicator = False

        for obs_id in all_obstacle_ids:
            contact_points = p.getContactPoints(car_id, obs_id)
            if len(contact_points) > 0:
                collided_indicator = True
                break
        
        if collided_indicator is True:
            print(f"*** At {step} step, (t={step*dt:.1f}s) Collision occurred! ***")
            break
        
        # Check goal reached
        dist_to_goal = math.hypot(car_state[0] - goal[0], car_state[1] - goal[1])
        if dist_to_goal < goal_tolerance:
            print(f"*** At {step} step, (t={step*dt:.1f}s) Goal reached! ***")
            time.sleep(2)
            break

        # pybullet setup (extra)
        p.stepSimulation()
        time.sleep(dt/2)
        
    # End of simulation loop
    print("Simulation ended.")
    p.disconnect()

if __name__ == "__main__":
    __main__()
    






