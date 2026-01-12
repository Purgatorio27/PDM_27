# Config file
import numpy as np

# Vehicle dimensions
vehicle_length = 1.5  # meters
vehicle_width = 1.0   # meters
lf = vehicle_length * 0.5  # distance from CG to front axle
lr = vehicle_length * 0.5  # distance from CG to rear axle
vehicle_radius = np.sqrt((vehicle_length / 2) ** 2 + (vehicle_width / 2) ** 2) 

# Dynamic constraints
max_speed = 4.0  # increased for faster response
min_speed = 0.3  # minimum forward speed to prevent complete stop
max_acceleration = 3.0  # increased for faster acceleration
max_deceleration = -3.0  # increased for faster braking
max_steering_angle = np.radians(45)  # increased for sharper turns

# Goal parameters
goal = (40, 40)

# Simulation time parameters
controller_dt = 0.05  # controller time step (50Hz = 0.02s, but user requested 0.05s)
sim_dt = 0.01  # fine-grained simulation dt for obstacle trajectories
sim_speed = 1.0  # obstacle speed multiplier (obstacles move this much faster)

dt = controller_dt  # for backward compatibility
max_steps = 900  # maximum number of simulation steps
T = controller_dt * max_steps  # total simulation time in seconds

# MPC parameters
mpc_horizon = 60  # prediction horizon steps
mpc_dt = 0.25  # MPC time step (larger than controller_dt for longer lookahead)
# Total lookahead = mpc_horizon * mpc_dt = 40 * 0.25 = 10 seconds = 40m at 4m/s


















