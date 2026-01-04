import numpy as np

# Function to generate dynamic obstacle trajectory
def generate_obs_trajectory(start_pos, velocity_reference, num_steps, dt):
    """
    Generate the trajectory of a dynamic obstacle over time, regarding variance of velocity.
    """
    
    trajectory = []

    v = np.array(velocity_reference)
    x, y = start_pos

    sigma = 0.4 * np.abs(velocity_reference)   # Standard deviation for Gaussian noise
    v = velocity_reference + np.random.randn(2) * sigma # Add Gaussian noise to velocity
    
    for _ in range(num_steps):
        
        # t = step * dt
        x = x + v[0] * dt
        y = y + v[1] * dt
        
        trajectory.append([x, y, v[0], v[1]])

    return np.array(trajectory)
