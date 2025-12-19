import numpy as np
import math


class RRTStar:
    """
    Docstring for RRTStar [TO DO]
    """
    def __init__(self, start, goal, model, env):
        self.nodes = [start]
        self.goal_pos = goal
        self.model = model
        self.env = env
        self.parents = {0: None}
        self.step_size = 0.5
        self.search_radius = 2.5
        self.car_radius = 0.4  # Safety margin around obstacles

    def collision_checker(self, x, y):
        """
        Docstring for collision_checker [TO DO]
        
        :param self: Description
        :param x: Description
        :param y: Description
        """
        if abs(x) > 4.8 or abs(y) > 4.8: return False  # Stay within walls

        # Check if there is no collision for all obstacles
        for obs in self.env.obstacles:
            ox, oy = obs['pos']
            dw, dh = obs['dim'][0]/2 + self.car_radius, obs['dim'][1]/2 + self.car_radius
            if abs(x - ox) < dw and abs(y - oy) < dh:
                return False
        return True

    def plan(self, max_iter=10000):
        """
        Docstring for plan [TO DO]
        
        :param self: Description
        :param max_iter: Description
        """
        for i in range(max_iter):
            if np.random.rand() < 0.15:  # 15% bias towards goal, 85% random
                sample = self.goal_pos
            else:
                sample = (np.random.uniform(-4.5, 4.5), np.random.uniform(-4.5, 4.5))
            
            dists = [math.hypot(n.x - sample[0], n.y - sample[1]) for n in self.nodes]
            nearest_node_idx = np.argmin(dists)  # Closest neighbour

            new_node = self.model.next_state(self.nodes[nearest_node_idx], sample, self.step_size)
            
            if new_node and self.collision_checker(new_node.x, new_node.y):  # Check if path is collsion free
                # Find all neighbours within search radius
                nearby = [j for j, n in enumerate(self.nodes) if math.hypot(n.x - new_node.x, n.y - new_node.y) < self.search_radius]
                best_idx = nearest_node_idx
                min_cost = self.nodes[nearest_node_idx].cost + self.step_size

                # Choose lowest cost parent
                for neighbor_idx in nearby:
                    reached_state = self.model.next_state(self.nodes[neighbor_idx], (new_node.x, new_node.y), self.step_size)

                    # Check if kinematically feasible
                    if reached_state and math.hypot(reached_state.x - new_node.x, reached_state.y - new_node.y) < 0.1:
                        if self.nodes[neighbor_idx].cost + self.step_size < min_cost:  # Check if cost is lower if we change parent and change accordingly
                            best_idx = neighbor_idx
                            min_cost = self.nodes[neighbor_idx].cost + self.step_size

                # Add to tree            
                new_node.cost = min_cost
                self.nodes.append(new_node)
                self.parents[len(self.nodes)-1] = best_idx
                
        goal_dists = [math.hypot(n.x - self.goal_pos[0], n.y - self.goal_pos[1]) for n in self.nodes]
        return self._extract_path(np.argmin(goal_dists)) if min(goal_dists) < 1.5 else None  # Goal reached if we are within 1.5m

    def _extract_path(self, idx):
        path = []
        while idx is not None:
            path.append(self.nodes[idx]); idx = self.parents[idx]
        return path[::-1]
