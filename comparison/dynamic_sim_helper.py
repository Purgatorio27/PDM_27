
import numpy as np
import random
from shared_config import DYNAMIC_OBSTACLE_COUNT, DYNAMIC_OBSTACLE_RADIUS, MAP_SIZE, STATIC_OBSTACLES

class DynamicObstacleManager:
    def __init__(self, count=DYNAMIC_OBSTACLE_COUNT, radius=DYNAMIC_OBSTACLE_RADIUS, seed=42):
        self.count = count
        self.radius = radius
        self.map_size = MAP_SIZE
        self.obstacles = []
        
        # Determine obstacle generation region (avoid start zone)
        self.safe_margin = 5.0
        
        # Initialize obstacles with random positions and velocities
        np.random.seed(seed)
        random.seed(seed)
        
        for _ in range(self.count):
            while True:
                # Random position
                x = np.random.uniform(self.safe_margin, self.map_size - 1)
                y = np.random.uniform(self.safe_margin, self.map_size - 1)
                
                # Check collision with static
                valid = True
                for s_obs in STATIC_OBSTACLES:
                    dist = np.hypot(x - s_obs['position'][0], y - s_obs['position'][1])
                    if dist < (s_obs['radius'] + self.radius + 0.5):
                        valid = False
                        break
                
                if valid:
                    # Random velocity (approx 1 m/s)
                    angle = np.random.uniform(0, 2 * np.pi)
                    speed = np.random.uniform(0.5, 1.5)
                    vx = speed * np.cos(angle)
                    vy = speed * np.sin(angle)
                    
                    self.obstacles.append({
                        'x': x,
                        'y': y,
                        'vx': vx,
                        'vy': vy,
                        'radius': self.radius,
                        'id': _
                    })
                    break

    def update(self, dt):
        """Update obstacle positions and bounce off walls."""
        for obs in self.obstacles:
            # Update position
            obs['x'] += obs['vx'] * dt
            obs['y'] += obs['vy'] * dt
            
            # Boundary checks (simple bounce)
            if obs['x'] < 0.0 + obs['radius']:
                obs['x'] = 0.0 + obs['radius']
                obs['vx'] *= -1
            elif obs['x'] > self.map_size - obs['radius']:
                obs['x'] = self.map_size - obs['radius']
                obs['vx'] *= -1
                
            if obs['y'] < 0.0 + obs['radius']:
                obs['y'] = 0.0 + obs['radius']
                obs['vy'] *= -1
            elif obs['y'] > self.map_size - obs['radius']:
                obs['y'] = self.map_size - obs['radius']
                obs['vy'] *= -1

    def get_for_mpc(self):
        """
        Return in format expected by MPC_core.
        MPC_core expects: [{'trajectory': (x, y, vx, vy), 'radius': r}]
        """
        mpc_obs = []
        for obs in self.obstacles:
            mpc_obs.append({
                'trajectory': (obs['x'], obs['y'], obs['vx'], obs['vy']),
                'radius': obs['radius']
            })
        return mpc_obs

    def get_for_log(self):
        """Return dict for logging."""
        return [{'id': o['id'], 'x': o['x'], 'y': o['y'], 'vx': o['vx'], 'vy': o['vy'], 'radius': o['radius']} for o in self.obstacles]
