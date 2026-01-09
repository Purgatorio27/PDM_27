import pybullet as p
import pybullet_data
import numpy as np

gray = [0.6, 0.6, 0.6, 1]
red = [0.8, 0.2, 0.2, 1]
blue = [0.2, 0.2, 0.8, 1]
green = [0.1, 0.7, 0.1, 1]


class Environment:
    def __init__(self, randomize=False):
        if not p.isConnected(): 
            p.connect(p.GUI)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.loadURDF("plane.urdf")
        p.resetDebugVisualizerCamera(12.0, 0, -89.9, [0, 0, 0])
        
        self.obstacles = []  # List of {'id', 'position', 'type', 'radius', 'color'}
        self._id_counter = 0

        # Walls
        self._add_rectangular_object(-5.0, 5.0, 5.0, 'h', 0.2, gray)
        self._add_rectangular_object(-5.0, 5.0, -5.0, 'h', 0.2, gray)
        self._add_rectangular_object(-5.0, 5.0, -5.0, 'v', 0.2, gray)
        self._add_rectangular_object(-5.0, 5.0, 5.0, 'v', 0.2, gray)

        if randomize:
            self._add_random_content(max_obstacles=8)
        else:
            self._setup_workshop_content()

        self._build_world()

    def _get_next_id(self):
        self._id_counter += 1
        return self._id_counter

    def _add_random_content(self, max_obstacles=6):
        """Generates workshop clusters while ensuring no overlaps."""        
        # Safe zones for Start [-4, -4] and Goal [4, 4]
        safe_zones = [np.array([-4.2, -4.2]), np.array([4.2, 4.2])]
        
        clusters_placed = 0
        attempts = 0

        # Keep track of center positions for overlap checking
        placed_centers = []

        while clusters_placed < max_obstacles and attempts < 100:
            attempts += 1
            cx = np.random.uniform(-3.5, 3.5)
            cy = np.random.uniform(-3.5, 3.5)
            new_pos = np.array([cx, cy])

            too_close = False
            # Check against start/goal
            for zone in safe_zones:
                if np.linalg.norm(new_pos - zone) < 2.0: 
                    too_close = True
                    break
            if too_close: continue
            
            # Check against other randomized clusters
            for old_pos in placed_centers:
                if np.linalg.norm(new_pos - old_pos) < 2.5:
                    too_close = True
                    break
            if too_close: continue
            
            # If safe, pick a template and use _add_workshop_item
            template = np.random.choice(['car', 'storage', 'clutter'])
            
            if template == 'car':
                self._add_workshop_item([cx, cy], [1.2, 2.2], blue)
                self._add_workshop_item([cx + 0.9, cy], [0.4, 0.4], red)
            
            elif template == 'storage':
                is_v = np.random.choice([True, False])
                dim = [0.4, 2.0] if is_v else [2.0, 0.4]
                self._add_workshop_item([cx, cy], dim, gray)
            
            elif template == 'clutter':
                self._add_workshop_item([cx, cy], [0.8, 0.8], green)
                self._add_workshop_item([cx - 0.7, cy + 0.7], [0.4, 0.4], green)
            
            placed_centers.append(new_pos)
            clusters_placed += 1

    def _add_rectangular_object(self, start, end, fixed, orientation, thickness, color):
        """Fills a line with squares, insetting centers so edges stay within bounds."""
        radius = thickness / 2
        
        # Inset centers by radius to make the edges flush with start/end
        adj_start = start + radius
        adj_end = end - radius
        
        length = abs(adj_end - adj_start)
        
        step_target = thickness * 0.8  # Calculate steps (80% thickness overlap for smooth distance fields)
        num_points = max(1, int(np.ceil(length / step_target)) + 1)
        
        # Generate centers precisely within the inset range
        if length < 1e-5:
            points = [adj_start]
        else:
            points = np.linspace(adj_start, adj_end, num_points)
        
        for p_val in points:
            pos = (float(p_val), float(fixed)) if orientation == 'h' else (float(fixed), float(p_val))
            self.obstacles.append({
                'id': self._get_next_id(),
                'position': pos,
                'type': 'square',
                'radius': radius,
                'color': color
            })

    def _add_workshop_item(self, pos, dim, color):
        """Converts box dimensions into differentiable square points with inset logic: start inserting object from the radius"""
        cx, cy = pos
        w, h = dim
        
        if abs(w - h) < 1e-3:
            self.obstacles.append({
                'id': self._get_next_id(),
                'position': (float(cx), float(cy)),
                'type': 'square',
                'radius': w / 2,
                'color': color
            })
        elif w > h:
            self._add_rectangular_object(cx - w/2, cx + w/2, cy, 'h', h, color)
        else:
            self._add_rectangular_object(cy - h/2, cy + h/2, cx, 'v', w, color)

    def _setup_workshop_content(self):
        """The original workshop layout from RRT*"""
        items = [
            {'pos': [-2.5, 3.0], 'dim': [1.2, 2.5], 'color': blue},
            {'pos': [-1.6, 2.2], 'dim': [0.4, 0.4], 'color': red},
            {'pos': [-3.0, 4.5], 'dim': [1.5, 0.4], 'color': gray},
            {'pos': [2.5, -3.0], 'dim': [1.2, 2.5], 'color': blue},
            {'pos': [1.6, -2.2], 'dim': [0.4, 0.4], 'color': red},
            {'pos': [3.0, -4.5], 'dim': [1.5, 0.4], 'color': gray},
            {'pos': [0.0, 0.0], 'dim': [2.0, 0.6], 'color': gray},
            {'pos': [1.3, 0.0], 'dim': [0.4, 0.4], 'color': red},
            {'pos': [-2.3, -2.0], 'dim': [1.0, 1.0], 'color': green}, 
            {'pos': [3.5, 1.0], 'dim': [0.6, 0.8], 'color': green},
            {'pos': [1.5, 3.5], 'dim': [0.7, 0.7], 'color': green},
            {'pos': [-4.5, 1.0], 'dim': [0.4, 2.5], 'color': gray},
            {'pos': [4.5, -1.0], 'dim': [0.4, 2.5], 'color': gray}
        ]
        for item in items:
            self._add_workshop_item(item['pos'], item['dim'], item['color'])

    def _build_world(self):
        for obs in self.obstacles:
            r = obs['radius']
            pos = [obs['position'][0], obs['position'][1], 0.5]
            
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[r, r, 0.5])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[r, r, 0.5], rgbaColor=obs['color'])
            
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos
            )

    def spawn_car(self, x, y):
        return p.createMultiBody(1.0, 
            p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.1]), 
            p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.1], rgbaColor=[0.1, 0.3, 0.9, 1]),
            [x, y, 0.15])
