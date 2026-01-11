"""
Shared configuration for MPC vs RRT* comparison experiments.
Both systems import from here to ensure identical environments.
"""
import numpy as np

# ============== Environment Settings ==============
MAP_SIZE = 30.0
START_POS = (2.0, 2.0)
GOAL_POS = (26.0, 26.0)
START_YAW = np.radians(45)

# ============== Vehicle Parameters ==============
# Match MPC Config.py values for consistency
VEHICLE_LENGTH = 1.5  # meters
VEHICLE_WIDTH = 1.0   # meters
VEHICLE_RADIUS = 0.9015  # sqrt((1.5/2)^2 + (1.0/2)^2) - matches MPC

# ============== Obstacle Settings ==============
DYNAMIC_OBSTACLE_COUNT = 8
DYNAMIC_OBSTACLE_RADIUS = 1.0

# Format: {'position': [x, y], 'radius': r}
STATIC_OBSTACLES = [
    {'position': [8.0, 8.0], 'radius': 1.2},
    {'position': [12.0, 12.0], 'radius': 1.5},
    {'position': [17.0, 17.0], 'radius': 1.2},
    {'position': [5.0, 12.0], 'radius': 1.0},
    {'position': [4.0, 18.0], 'radius': 1.3},
    {'position': [18.0, 6.0], 'radius': 1.0},
    {'position': [20.0, 12.0], 'radius': 1.2},
    {'position': [10.0, 5.0], 'radius': 0.8},
    {'position': [15.0, 9.0], 'radius': 1.0},
    {'position': [9.0, 16.0], 'radius': 0.9},
    {'position': [14.0, 20.0], 'radius': 1.1},
    {'position': [22.0, 22.0], 'radius': 1.4},
    {'position': [25.0, 15.0], 'radius': 1.2},
    {'position': [15.0, 25.0], 'radius': 1.3},
]

# ============== Simulation Parameters ==============
MAX_SPEED = 4.0  # m/s
MIN_SPEED = 0.3  # m/s
GOAL_TOLERANCE = 0.5  # meters

# ============== Helper Functions ==============
def get_obstacles_for_rrt():
    """Convert obstacles to RRT* format (box approximation)."""
    obstacles = []
    for obs in STATIC_OBSTACLES:
        obstacles.append({
            'type': 'circle',
            'pos': obs['position'],
            'radius': obs['radius']
        })
    return obstacles

def get_obstacles_for_mpc():
    """Return obstacles in MPC format."""
    obstacles = []
    for obs in STATIC_OBSTACLES:
        obstacles.append({
            'position': obs['position'],
            'type': 'circle',
            'radius': obs['radius']
        })
    return obstacles

def print_config():
    """Print current configuration."""
    print(f"Map size: {MAP_SIZE}x{MAP_SIZE}")
    print(f"Start: {START_POS}, Goal: {GOAL_POS}")
    print(f"Static obstacles: {len(STATIC_OBSTACLES)}")
    for i, obs in enumerate(STATIC_OBSTACLES):
        print(f"  {i+1}. pos={obs['position']}, r={obs['radius']}")
