import pybullet as p
import pybullet_data
import numpy as np


gray = [0.6, 0.6, 0.6, 1]
red = [0.8, 0.2, 0.2, 1]
blue = [0.2, 0.2, 0.8, 1]
green = [0.1, 0.7, 0.1, 1]


class Environment:
    """
    Docstring for Environment [TO DO]
    """
    def __init__(self, randomize=False):
        if not p.isConnected(): p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.loadURDF("plane.urdf")
        p.resetDebugVisualizerCamera(12.0, 0, -89.9, [0, 0, 0]) # Top view
        
        self.obstacles = []

        # Walls
        self.obstacles.append({'type': 'box', 'pos': [0, 5], 'dim': [10, 0.2], 'color': gray})
        self.obstacles.append({'type': 'box', 'pos': [0, -5], 'dim': [10, 0.2], 'color': gray})
        self.obstacles.append({'type': 'box', 'pos': [5, 0], 'dim': [0.2, 10], 'color': gray})
        self.obstacles.append({'type': 'box', 'pos': [-5, 0], 'dim': [0.2, 10], 'color': gray})
        
        if randomize:
            self._add_random_content(max_obstacles=6)
        else:
            self._add_static_content()
            
        self._build_world()

    # def _add_static_content(self):
    #     """The hardcoded layout with 2 walls and 1 obstacle."""
    #     self.obstacles.extend([
    #         {'type': 'box', 'pos': [0, 0], 'dim': [1.0, 1.0], 'color': red},
    #         {'type': 'box', 'pos': [-2.5, 2.2], 'dim': [5.0, 0.2], 'color': gray},
    #         {'type': 'box', 'pos': [2.5, -2.2], 'dim': [5.0, 0.2], 'color': gray}
    #     ])

    def _add_static_content(self):
        """Workshop layout for cars and mechanics instead of factory :D"""        
        self.obstacles.extend([
            # Top left car with mechanic and rack
            {'type': 'box', 'pos': [-2.5, 3.0], 'dim': [1.2, 2.5], 'color': blue},
            {'type': 'box', 'pos': [-1.6, 2.2], 'dim': [0.4, 0.4], 'color': red},
            {'type': 'box', 'pos': [-3.0, 4.5], 'dim': [1.5, 0.4], 'color': gray},

            # Bottom right car with mechanic and rack
            {'type': 'box', 'pos': [2.5, -3.0], 'dim': [1.2, 2.5], 'color': blue},
            {'type': 'box', 'pos': [1.6, -2.2], 'dim': [0.4, 0.4], 'color': red},
            {'type': 'box', 'pos': [3.0, -4.5], 'dim': [1.5, 0.4], 'color': gray},

            # Middle wall
            {'type': 'box', 'pos': [0.0, 0.0], 'dim': [2.0, 0.6], 'color': gray},
            {'type': 'box', 'pos': [1.3, 0.0], 'dim': [0.4, 0.4], 'color': red},
            
            # Random obstacles
            {'type': 'box', 'pos': [-2.3, -2.0], 'dim': [1.0, 1.0], 'color': green}, 
            {'type': 'box', 'pos': [3.5, 1.0], 'dim': [0.6, 0.8], 'color': green},
            {'type': 'box', 'pos': [1.5, 3.5], 'dim': [0.7, 0.7], 'color': green},

            # Storage on the sides
            {'type': 'box', 'pos': [-4.5, 1.0], 'dim': [0.4, 2.5], 'color': gray},
            {'type': 'box', 'pos': [4.5, -1.0], 'dim': [0.4, 2.5], 'color': gray}
        ])

    def _add_random_content(self, max_obstacles=6):
        """Generates workshop clusters while ensuring no overlaps."""        
        safe_zones = [np.array([-4.2, -4.2]), np.array([4.2, 4.2])]  # No obstacles on start and goal
        
        clusters_placed = 0
        attempts = 0

        while clusters_placed < max_obstacles and attempts < 100:
            attempts += 1
            cx = np.random.uniform(-3.5, 3.5)
            cy = np.random.uniform(-3.5, 3.5)
            new_pos = np.array([cx, cy])

            # Check distance against start/goal
            too_close = False
            for zone in safe_zones:
                if np.linalg.norm(new_pos - zone) < 1.8: 
                    too_close = True
                    break
            if too_close: continue
            
            # Check distance against existing obstacles to prevent overlap
            for obs in self.obstacles:
                obs_pos = np.array([obs['pos'][0], obs['pos'][1]])
                if np.linalg.norm(new_pos - obs_pos) < 2.2:
                    too_close = True
                    break
            
            if not too_close:
                template = np.random.choice(['car', 'storage', 'clutter'])
                
                if template == 'car':
                    # Car and mechanic
                    self.obstacles.append({'type': 'box', 'pos': [cx, cy], 'dim': [1.2, 2.2], 'color': blue})
                    self.obstacles.append({'type': 'box', 'pos': [cx + 0.8, cy], 'dim': [0.4, 0.4], 'color': red})
                
                elif template == 'storage':
                    # Vertical or horizontal rack
                    is_v = np.random.choice([True, False])
                    dim = [0.4, 2.0] if is_v else [2.0, 0.4]
                    self.obstacles.append({'type': 'box', 'pos': [cx, cy], 'dim': dim, 'color': gray})
                
                elif template == 'clutter':
                    # Toolboxes grouped together
                    self.obstacles.append({'type': 'box', 'pos': [cx, cy], 'dim': [0.8, 0.8], 'color': green})
                    self.obstacles.append({'type': 'box', 'pos': [cx - 0.6, cy + 0.6], 'dim': [0.4, 0.4], 'color': green})
                
                clusters_placed += 1

    def _build_world(self):
        for obs in self.obstacles:
            half_ext = [obs['dim'][0]/2, obs['dim'][1]/2, 0.5]

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=obs['color']),
                basePosition=[obs['pos'][0], obs['pos'][1], 0.5]
            )

    def spawn_car(self, start_state):
        return p.createMultiBody(1.0, 
            p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.1]), 
            p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.1], rgbaColor=[0.1, 0.3, 0.9, 1]),
            [start_state.x, start_state.y, 0.15])
