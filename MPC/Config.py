# Config file
import numpy as np

# Vehicle dimensions
vehicle_length = 1.5  # meters
vehicle_width = 1.0   # meters
lf = vehicle_length * 0.5  # distance from CG to front axle
lr = vehicle_length * 0.5  # distance from CG to rear axle
vehicle_radius = np.sqrt((vehicle_length / 2) ** 2 + (vehicle_width / 2) ** 2)

# Dynamic constraints
max_speed = 2.5
min_speed = -1.0
max_acceleration = 1.0
max_deceleration = -1.5
max_steering_angle = np.radians(30)  # radians

# Goal parameters
goal = (20, 20)

# Simulation time parameters
dt = 0.1  # time step
max_steps = 5000  # maximum number of simulation steps
T = dt * max_steps  # total simulation time in seconds


















