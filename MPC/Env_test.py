import pybullet as p
import pybullet_data
import time
import numpy as np
import math

# Import existing modules
from Obstacles import static_obstacles, dynamic_obstacles
from Config import T, max_steps, dt, goal, lf, lr, vehicle_length, vehicle_width, vehicle_radius

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
    visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=d_obs['radius'], length = 0.2, rgbaColor=[1, 1, 0, 0.8])  # 黄色动态圆
    
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=initial_pos
    )
    dyn_body_ids.append(body_id)

all_obstacle_ids = stat_body_ids + dyn_body_ids

def __main__():

    for step in range(max_steps):
        # Update dynamic obstacles positions in PyBullet
        for i, dyn_id in enumerate(dyn_body_ids):
            current_pos_2d = dynamic_obstacles[i]['trajectory'][step][0:2]
            pos_3d = current_pos_2d.tolist() + [0.1]
            p.resetBasePositionAndOrientation(dyn_id, pos_3d, [0, 0, 0, 1])
    

        # pybullet setup (extra)
        p.stepSimulation()
        time.sleep(dt/2)
        
    print("Simulation loop finished. Window will stay open. Close the window to exit...")
    
    while p.isConnected(physicsClient):
        p.stepSimulation()
        time.sleep(0.01) 

if __name__ == "__main__":
    __main__()