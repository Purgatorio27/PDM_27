import math
from dataclasses import dataclass


@dataclass
class State:
    """
    Attributes:
        x: Global x-coordinate in meters.
        y: Global y-coordinate in meters.
        yaw: Heading angle in radians.
        cost: Cumulative path distance from the start node.
    """
    x: float
    y: float
    yaw: float
    # v: float
    cost: float = 0.0


# class State:
#     def __init__(self, x, y, yaw, cost=0.0):
#         self.x = x
#         self.y = y
#         self.yaw = yaw
#         self.cost = cost


class KinematicBicycleModelRRT:
    """
    A simplified bicycle model assuming that there is no slip, therefore ignoring beta.
    """
    def __init__(self, min_radius):
        """
        Args:
            L: Distance between front and rear axles
            min_radius: Minimum turning radius based on max steering angle.
        """
        self.min_radius = min_radius

    def next_state(self, state, target_pos, step_size):
        """
        Computes a new state by following a circular arc a certain step_size moves along towards a target position.
        
        Args:
            state: The current vehicle state.
            target_pos: The (x, y) coordinates of the target position.
            step_size: The distance to travel along the arc.
        """
        dx = target_pos[0] - state.x
        dy = target_pos[1] - state.y

        # Translation to vehicles egocentric frame (rotation matrix with angle yaw)
        lx = dx * math.cos(state.yaw) + dy * math.sin(state.yaw)
        ly = -dx * math.sin(state.yaw) + dy * math.cos(state.yaw)

        if lx <= 0: return None  # Ensure that the car only moves forward with respect to itself

        # radius = (lx**2 + ly**2) / (2 * ly)  # We can't do this, because it would create a division by zero
        radius = float('inf') if abs(ly) < 0.01 else (lx**2 + ly**2) / (2 * ly)  # Move straight  if the y distance between current position and target position is very close 

        if abs(radius) < self.min_radius: return None  # Prevent turns that are too sharp
        
        if radius == float('inf'):  # Moving straight
            new_x = state.x + step_size * math.cos(state.yaw)
            new_y = state.y + step_size * math.sin(state.yaw)
            new_yaw = state.yaw
        else:
            angle_step = step_size / radius
            new_yaw = state.yaw + angle_step
            new_x = state.x + radius * (math.sin(new_yaw) - math.sin(state.yaw))
            new_y = state.y - radius * (math.cos(new_yaw) - math.cos(state.yaw))

        return State(new_x, new_y, new_yaw)
    
class SimpleHolonomicModel:
    def next_state(self, state, target_pos, step_size):
        dx = target_pos[0] - state.x
        dy = target_pos[1] - state.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6: return state
        new_yaw = math.atan2(dy, dx)
        new_x = state.x + step_size * (dx / dist)
        new_y = state.y + step_size * (dy / dist)
        return State(new_x, new_y, new_yaw)
