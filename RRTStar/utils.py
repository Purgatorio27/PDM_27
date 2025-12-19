import numpy as np
import math
from scipy.interpolate import make_interp_spline


def generate_trajectory(planner, smoothing_factor, target_velocity, max_attempts=5, max_iter=15000):
    """
    Docstring for generate_trajectory [TO DO]
    
    :param planner: Description
    :param smoothing_factor: Description
    :param target_velocity: Description
    :param max_attempts: Description
    :param max_iter: Description
    """
    rrt_path = None

    for attempt in range(max_attempts):  # Loop if RRT* could not find a path the first iterations, more likely to get a path
        print(f"Generating path (Attempt {attempt + 1}/{max_attempts})")
        rrt_path = planner.plan(max_iter=max_iter)
        
        if rrt_path is not None:
            print("Path found")
            break
    
    if rrt_path is None:
        print(f"Path not found")
        return None, None

    smoothed_traj = smooth_trajectory(
        rrt_path, 
        smoothing_factor=smoothing_factor, 
        target_velocity=target_velocity
    )

    return rrt_path, smoothed_traj


def smooth_trajectory(RRT_path, smoothing_factor, target_velocity):
    """
    Converts RRT* trajectory into a smooth trajectory.
    """
    points = np.array([[n.x, n.y] for n in RRT_path])
     
    t = np.linspace(0, 1, len(points))
    t_smooth = np.linspace(0, 1, len(points) * smoothing_factor)
    
    # Cubic spline interpolation   
    spline_x = make_interp_spline(t, points[:, 0], k=3)(t_smooth)
    spline_y = make_interp_spline(t, points[:, 1], k=3)(t_smooth)

    # Calculate yaw and velocity
    trajectory = []
    for i in range(len(spline_x)):
        # Calculate heading based on the next point
        if i < len(spline_x) - 1:
            dy = spline_y[i+1] - spline_y[i]
            dx = spline_x[i+1] - spline_x[i]
            yaw = math.atan2(dy, dx)
        else:
            yaw = trajectory[-1][2]  # Use the previous heading for the last point
            
        trajectory.append([spline_x[i], spline_y[i], yaw, target_velocity])

    return np.array(trajectory)
