"""
MPC_RRT: Integrated RRT* Global Path Planning with MPC Local Control

This module combines:
1. RRT* for global path planning - finds collision-free path through obstacles
2. MPC for local trajectory tracking - smooth control with real-time obstacle avoidance

The RRT* path provides waypoints that the MPC tracks, while MPC maintains
smooth dynamics and handles any dynamic obstacles or small perturbations.

Architecture:
- RRT* plans a global path avoiding static obstacles
- MPC optimizes control to reach the final goal
- RRT path is used for warm-start initialization of MPC trajectory
- Deviation detection triggers replanning when vehicle strays from path

Limitations:
- MPC uses soft constraints for obstacle avoidance (not guaranteed collision-free)
- MPC goal is set at compile time; dynamic goal changes require solver recompilation
- In tight obstacle configurations, MPC may clip obstacles slightly

Usage:
    from MPC_RRT import MPC_RRT_Controller, create_mpc_rrt_controller
    
    controller = create_mpc_rrt_controller(
        start_state=[x, y, v, psi],
        goal=GOAL_POS,
        static_obstacles=obstacles,
        map_size=25.0
    )
    
    # In control loop:
    control, status, info = controller.solve(current_state)
"""

import numpy as np
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# MPC imports
from MPC_core import MPC
import Config as cfg

# RRT imports - need to handle path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RRTStar.KinematicBicycleModelRRT import KinematicBicycleModelRRT, State


@dataclass
class Waypoint:
    """A waypoint along the RRT path."""
    x: float
    y: float
    yaw: float
    index: int


class RRTPlanner:
    """
    RRT* planner adapted for integration with MPC.
    Generates global path from start to goal avoiding static obstacles.
    """
    
    def __init__(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float],
                 static_obstacles: List[Dict], map_size: float = 25.0,
                 vehicle_radius: float = 0.9, safety_margin: float = 0.3):
        """
        Initialize RRT* planner.
        
        Args:
            start_pos: (x, y) start position
            goal_pos: (x, y) goal position
            static_obstacles: List of {'position': [x, y], 'radius': r}
            map_size: Size of the square map
            vehicle_radius: Vehicle bounding radius
            safety_margin: Extra safety margin around obstacles
        """
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.static_obstacles = static_obstacles
        self.map_size = map_size
        self.vehicle_radius = vehicle_radius
        self.safety_margin = safety_margin
        self.car_radius = vehicle_radius + safety_margin
        
        # RRT* parameters
        self.step_size = 0.8  # Step size for tree expansion
        self.search_radius = 3.0  # Radius for rewiring
        self.goal_bias = 0.15  # Probability of sampling goal
        self.max_iterations = 20000
        
        # Kinematic model for RRT
        self.min_turn_radius = 1.5  # Matches MPC vehicle model
        self.model = KinematicBicycleModelRRT(self.min_turn_radius)
        
        # Tree storage
        self.nodes: List[State] = []
        self.parents: Dict[int, Optional[int]] = {}
        
        # Planning metrics
        self.planning_time = 0
        self.iterations_used = 0
        self.path: Optional[List[State]] = None
    
    def collision_checker(self, x: float, y: float) -> bool:
        """
        Check if position is collision-free.
        
        Args:
            x, y: Position to check
            
        Returns:
            True if collision-free, False otherwise
        """
        # Boundary check
        margin = 0.5
        if x < margin or x > self.map_size - margin:
            return False
        if y < margin or y > self.map_size - margin:
            return False
        
        # Obstacle collision check
        for obs in self.static_obstacles:
            ox, oy = obs['position']
            obs_radius = obs['radius']
            dist = math.hypot(x - ox, y - oy)
            if dist < (obs_radius + self.car_radius):
                return False
        
        return True
    
    def path_collision_checker(self, x1: float, y1: float, x2: float, y2: float,
                                num_checks: int = 10) -> bool:
        """
        Check if straight line path between two points is collision-free.
        
        Args:
            x1, y1: Start point
            x2, y2: End point
            num_checks: Number of intermediate points to check
            
        Returns:
            True if path is collision-free, False otherwise
        """
        for i in range(num_checks + 1):
            t = i / num_checks
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self.collision_checker(x, y):
                return False
        return True
    
    def plan(self, max_iter: Optional[int] = None) -> Optional[List[State]]:
        """
        Plan path using RRT*.
        
        Args:
            max_iter: Maximum iterations (default: self.max_iterations)
            
        Returns:
            List of State objects forming the path, or None if planning fails
        """
        if max_iter is None:
            max_iter = self.max_iterations
        
        start_time = time.time()
        
        # Initialize tree with start state
        start_yaw = math.atan2(self.goal_pos[1] - self.start_pos[1],
                               self.goal_pos[0] - self.start_pos[0])
        start_state = State(self.start_pos[0], self.start_pos[1], start_yaw, 0.0)
        self.nodes = [start_state]
        self.parents = {0: None}
        
        best_goal_idx = None
        best_goal_dist = float('inf')
        
        for i in range(max_iter):
            self.iterations_used = i + 1
            
            # Sample with goal bias
            if np.random.rand() < self.goal_bias:
                sample = self.goal_pos
            else:
                sample = (
                    np.random.uniform(0.5, self.map_size - 0.5),
                    np.random.uniform(0.5, self.map_size - 0.5)
                )
            
            # Find nearest node
            dists = [math.hypot(n.x - sample[0], n.y - sample[1]) for n in self.nodes]
            nearest_idx = int(np.argmin(dists))
            
            # Extend toward sample using kinematic model
            new_node = self.model.next_state(self.nodes[nearest_idx], sample, self.step_size)
            
            if new_node and self.collision_checker(new_node.x, new_node.y):
                # Find nearby nodes for rewiring
                nearby = [j for j, n in enumerate(self.nodes)
                         if math.hypot(n.x - new_node.x, n.y - new_node.y) < self.search_radius]
                
                best_parent_idx = nearest_idx
                min_cost = self.nodes[nearest_idx].cost + self.step_size
                
                # Choose best parent from nearby nodes
                for neighbor_idx in nearby:
                    reached = self.model.next_state(
                        self.nodes[neighbor_idx],
                        (new_node.x, new_node.y),
                        self.step_size
                    )
                    if reached and math.hypot(reached.x - new_node.x, reached.y - new_node.y) < 0.1:
                        new_cost = self.nodes[neighbor_idx].cost + self.step_size
                        if new_cost < min_cost:
                            best_parent_idx = neighbor_idx
                            min_cost = new_cost
                
                # Add node to tree
                new_node.cost = min_cost
                new_idx = len(self.nodes)
                self.nodes.append(new_node)
                self.parents[new_idx] = best_parent_idx
                
                # Check if we reached goal region
                goal_dist = math.hypot(new_node.x - self.goal_pos[0],
                                       new_node.y - self.goal_pos[1])
                if goal_dist < best_goal_dist:
                    best_goal_dist = goal_dist
                    best_goal_idx = new_idx
                
                # Early termination if very close to goal
                if goal_dist < 1.0:
                    break
        
        self.planning_time = time.time() - start_time
        
        # Extract path if goal is reachable
        goal_threshold = 1.5  # meters
        if best_goal_dist < goal_threshold and best_goal_idx is not None:
            self.path = self._extract_path(best_goal_idx)
            return self.path
        
        print(f"[RRT] Planning failed. Best distance to goal: {best_goal_dist:.2f}m")
        return None
    
    def _extract_path(self, goal_idx: int) -> List[State]:
        """Extract path from tree by backtracking from goal."""
        path = []
        idx = goal_idx
        while idx is not None:
            path.append(self.nodes[idx])
            idx = self.parents[idx]
        return path[::-1]  # Reverse to get start-to-goal order
    
    def simplify_path(self, path: List[State], min_waypoint_dist: float = 2.0) -> List[State]:
        """
        Simplify path by removing unnecessary intermediate waypoints.
        Keeps waypoints at minimum distance apart and at direction changes.
        
        Args:
            path: Original path
            min_waypoint_dist: Minimum distance between waypoints
            
        Returns:
            Simplified path
        """
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        last_added_idx = 0
        
        for i in range(1, len(path) - 1):
            # Distance from last added waypoint
            dist_from_last = math.hypot(path[i].x - simplified[-1].x,
                                        path[i].y - simplified[-1].y)
            
            # Check if path from last waypoint to next is collision-free
            can_skip = self.path_collision_checker(
                simplified[-1].x, simplified[-1].y,
                path[i + 1].x, path[i + 1].y
            )
            
            # Keep waypoint if distance is significant or can't skip
            if dist_from_last >= min_waypoint_dist or not can_skip:
                simplified.append(path[i])
                last_added_idx = i
        
        # Always add final waypoint
        simplified.append(path[-1])
        
        return simplified


class MPC_RRT_Controller:
    """
    Integrated RRT* + MPC controller with pre-compiled solver pool.
    
    Strategy:
    1. RRT* plans global path from start to goal
    2. Identify key waypoints along the path
    3. PRE-COMPILE MPC solvers for each key waypoint (done during initialization)
    4. At runtime, simply SWITCH between pre-compiled solvers based on progress
    5. No runtime recompilation needed!
    """
    
    def __init__(self, start_state: List[float], goal: Tuple[float, float],
                 static_obstacles: List[Dict], dynamic_obstacles: List[Dict] = None,
                 map_size: float = 25.0, horizon: int = 40, dt: float = 0.2,
                 waypoint_tolerance: float = 2.0, lookahead_distance: float = 4.0,
                 replan_on_deviation: float = 8.0, num_precompiled_solvers: int = 15):
        """
        Initialize integrated RRT+MPC controller.
        
        Args:
            start_state: [x, y, v, psi] initial state
            goal: (x, y) goal position
            static_obstacles: List of static obstacles for MPC format
            dynamic_obstacles: List of dynamic obstacles (optional)
            map_size: Size of square map
            horizon: MPC prediction horizon
            dt: MPC time step
            waypoint_tolerance: Distance threshold to consider waypoint reached
            lookahead_distance: How far ahead to look for next waypoint
            replan_on_deviation: Deviation threshold to trigger replanning
            num_precompiled_solvers: Number of intermediate solvers to pre-compile
        """
        self.goal = goal
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles or []
        self.map_size = map_size
        self.horizon = horizon
        self.dt = dt
        
        # Waypoint tracking parameters
        self.waypoint_tolerance = waypoint_tolerance
        self.lookahead_distance = lookahead_distance
        self.replan_on_deviation = replan_on_deviation
        self.num_precompiled_solvers = num_precompiled_solvers
        
        # State tracking
        self.current_waypoint_idx = 0
        self.waypoints: List[Waypoint] = []
        self.rrt_path: Optional[List[State]] = None
        
        # Pre-compiled solver pool
        self.solver_pool: List[MPC] = []  # List of pre-compiled MPC solvers
        self.solver_targets: List[Tuple[float, float]] = []  # Target for each solver
        self.current_solver_idx = 0  # Index of currently active solver
        
        # Planning status
        self.planning_success = False
        self.planning_time = 0
        self.iterations_used = 0
        
        # Initialize RRT planner with small safety margin for path planning
        self.rrt_planner = RRTPlanner(
            start_pos=(start_state[0], start_state[1]),
            goal_pos=goal,
            static_obstacles=static_obstacles,
            map_size=map_size,
            vehicle_radius=cfg.vehicle_radius,
            safety_margin=0.5
        )
        
        # Plan initial path
        self._plan_rrt_path(start_state)
        
        # Pre-compile all MPC solvers for key waypoints
        self._precompile_solvers()
        
        # Set initial active solver
        self.mpc = self.solver_pool[0] if self.solver_pool else None
        self.current_mpc_target = self.solver_targets[0] if self.solver_targets else goal
        self.target_waypoint_idx = 0
        
        # Debug info
        self.debug_info = {}
    
    def _precompile_solvers(self):
        """
        Pre-compile MPC solvers for key waypoints along the RRT path.
        This is done once during initialization, so runtime is fast.
        """
        import sys
        from io import StringIO
        
        # Clear existing solvers
        self.solver_pool = []
        self.solver_targets = []
        self.solver_waypoint_indices = []  # Track which waypoint each solver targets
        
        if not self.waypoints or len(self.waypoints) < 2:
            # No waypoints, just compile one solver for the goal
            print(f"[MPC_RRT] Pre-compiling 1 solver for goal {self.goal}")
            self.solver_pool = [MPC(
                static_obstacles=self.static_obstacles,
                dynamic_obstacles=self.dynamic_obstacles,
                horizon=self.horizon,
                dt=self.dt,
                goal=self.goal,
                compile=True
            )]
            self.solver_targets = [self.goal]
            self.solver_waypoint_indices = [0]
            return
        
        # Select key waypoints for solver targets
        # Strategy: Ensure solvers are spaced roughly every 2.5 meters
        num_waypoints = len(self.waypoints)
        
        # Calculate spacing based on path length or density
        # Since waypoints are now interpolated to max 2.0m spacing, we can just pick every Nth one
        # But to be safe, we'll calculate cumulative distance
        
        cumulative_dist = [0.0]
        for i in range(1, len(self.waypoints)):
            wp_prev = self.waypoints[i-1]
            wp_curr = self.waypoints[i]
            d = math.hypot(wp_curr.x - wp_prev.x, wp_curr.y - wp_prev.y)
            cumulative_dist.append(cumulative_dist[-1] + d)
        
        total_length = cumulative_dist[-1]
        target_spacing = 1.5  # meters (reduced for tighter control)
        
        key_indices = []
        next_target_dist = target_spacing
        
        # Always include first target (index 1, skipping start 0)
        current_idx = 1
        
        while current_idx < num_waypoints:
            key_indices.append(current_idx)
            
            # Find next index that matches spacing
            target_dist = cumulative_dist[current_idx] + target_spacing
            
            # Scan forward
            while current_idx < num_waypoints and cumulative_dist[current_idx] < target_dist:
                current_idx += 1
            
            # If we overshot by a lot, maybe back up? No, simpler better.
            # If reached end, make sure to add the last one
        
        # Ensure final waypoint is included
        if key_indices[-1] != num_waypoints - 1:
            key_indices.append(num_waypoints - 1)
            
        print(f"[MPC_RRT] path_len={total_length:.1f}m. Pre-compiling {len(key_indices)} virtual solvers (spacing ~{target_spacing}m).")
        
        # Initialize single shared MPC solver if not exists
        if not hasattr(self, 'mpc') or self.mpc is None:
             print(f"[MPC_RRT] Initializing single shared MPC solver instance...")
             self.mpc = MPC(
                static_obstacles=self.static_obstacles,
                dynamic_obstacles=self.dynamic_obstacles,
                horizon=self.horizon,
                dt=self.dt,
                goal=self.goal,
                compile=True
             )

        for i, wp_idx in enumerate(key_indices):
            wp = self.waypoints[wp_idx]
            target = (wp.x, wp.y)
            
            # Use final goal for last solver (ensure we reach exact goal)
            if wp_idx == num_waypoints - 1:
                target = self.goal
            
            # Create lightweight instances that share the library
            # Compile=False for all these instances
            solver = MPC(
                static_obstacles=self.static_obstacles,
                dynamic_obstacles=self.dynamic_obstacles,
                horizon=self.horizon,
                dt=self.dt,
                goal=target,
                compile=False
            )
            
            self.solver_pool.append(solver)
            self.solver_targets.append(target)
            self.solver_waypoint_indices.append(wp_idx)
        
        print(f"[MPC_RRT] 'Pre-compiled' {len(self.solver_pool)} virtual solvers (Shared Instance).")
        for i, (target, wp_idx) in enumerate(zip(self.solver_targets, self.solver_waypoint_indices)):
            print(f"  Solver {i}: waypoint {wp_idx} -> target=({target[0]:.1f}, {target[1]:.1f})")
    
    def _select_solver(self, cur_state: List[float]) -> int:
        """
        Select the appropriate pre-compiled solver based on current position relative to path.
        
        Robust Strategy: 
        1. Find closest point on RRT path
        2. Select a target solver that is strictly AHEAD of current position
        3. Ensure we don't jump too far ahead or back
        
        Args:
            cur_state: [x, y, v, psi] current state
            
        Returns:
            Index of the solver to use
        """
        if len(self.solver_pool) <= 1:
            return 0
        
        x, y = cur_state[0], cur_state[1]
        
        # 1. Find closest waypoint on path (Robust localization)
        min_dist = float('inf')
        closest_wp_idx = 0
        for i, wp in enumerate(self.waypoints):
            d = math.hypot(x - wp.x, y - wp.y)
            if d < min_dist:
                min_dist = d
                closest_wp_idx = i
        
        # 2. Find the best solver target
        # We want a target that is AHEAD of the closest waypoint
        # But not too far ahead.
        
        desired_wp_idx = closest_wp_idx + 1 # Aim for next waypoint at minimum
        
        # Find the first solver whose target waypoint index >= desired_wp_idx
        best_solver_idx = len(self.solver_pool) - 1 # Default to last
        
        for i, target_wp_idx in enumerate(self.solver_waypoint_indices):
            if target_wp_idx >= desired_wp_idx:
                best_solver_idx = i
                break
        
        # 3. Hysteresis/Stability Check
        # If the suggested solver is far from the current one (jump), be careful
        # But given we want to recover from getting stuck, we should trust the geometry
        # unless it's a huge jump backwards (which might happen if path loops, but here path is simple)
        
        # Determine if we stick to current or switch
        current_solver_idx = self.current_solver_idx if hasattr(self, 'current_solver_idx') else 0
        
        # If geometric calculation suggests a diff solver
        if best_solver_idx != current_solver_idx:
            # Allow moving forward freely
            if best_solver_idx > current_solver_idx:
                return best_solver_idx
            
            # Require significant evidence to move backward (prevent oscillation)
            # Only reset backward if we are clearly closer to the earlier segment
            # And far from current target
            
            current_target_wp_idx = self.solver_waypoint_indices[current_solver_idx]
            current_target = self.waypoints[current_target_wp_idx]
            dist_to_current_target = math.hypot(x - current_target.x, y - current_target.y)
            
            if dist_to_current_target > 5.0 and best_solver_idx < current_solver_idx:
                print(f"[MPC_RRT] Detected large deviation from target {current_solver_idx} ({dist_to_current_target:.1f}m). Resetting to {best_solver_idx}")
                return best_solver_idx
                
            return current_solver_idx
            
        return best_solver_idx
    
    def _plan_rrt_path(self, current_state: List[float]) -> bool:
        """
        Plan RRT* path from current position to goal.
        
        Args:
            current_state: [x, y, v, psi] current state
            
        Returns:
            True if planning successful, False otherwise
        """
        x, y = current_state[0], current_state[1]
        
        # Update RRT planner start position
        self.rrt_planner.start_pos = (x, y)
        
        # Plan path with retries
        path = None
        max_retries = 5
        for attempt in range(max_retries):
            path = self.rrt_planner.plan()
            if path is not None:
                if attempt > 0:
                    print(f"[MPC_RRT] Planning succeeded on attempt {attempt+1}")
                break
            print(f"[MPC_RRT] Planning attempt {attempt+1} failed. Retrying with new seed...")
            np.random.seed(int(time.time() * 1000) % 100000 + attempt)
        
        if path is None:
            print("[MPC_RRT] RRT planning failed after all retries!")
            self.planning_success = False
            return False
        
        # Simplify path
        simplified_path = self.rrt_planner.simplify_path(path, min_waypoint_dist=2.0)
        
        # Interpolate path to ensure waypoints are not too far apart (max 2.0m)
        # This ensures MPC always has a close enough target
        interpolated_path = []
        if len(simplified_path) > 0:
            interpolated_path.append(simplified_path[0])
            for i in range(len(simplified_path) - 1):
                p1 = simplified_path[i]
                p2 = simplified_path[i+1]
                dist = math.hypot(p2.x - p1.x, p2.y - p1.y)
                
                if dist > 2.0:
                    num_segments = math.ceil(dist / 2.0)
                    for j in range(1, num_segments):
                        t = j / num_segments
                        new_x = p1.x + t * (p2.x - p1.x)
                        new_y = p1.y + t * (p2.y - p1.y)
                        # Linear interpolation for yaw might be approximate but sufficient
                        new_yaw = p1.yaw + t * (p2.yaw - p1.yaw) 
                        interpolated_path.append(State(new_x, new_y, new_yaw, 0.0))
                
                interpolated_path.append(p2)
        
        # Convert to waypoints
        self.waypoints = []
        for i, state in enumerate(interpolated_path):
            wp = Waypoint(x=state.x, y=state.y, yaw=state.yaw, index=i)
            self.waypoints.append(wp)
        
        # Add goal as final waypoint if not close enough
        if len(self.waypoints) > 0:
            last_wp = self.waypoints[-1]
            goal_dist = math.hypot(last_wp.x - self.goal[0], last_wp.y - self.goal[1])
            if goal_dist > 0.5:
                goal_yaw = math.atan2(self.goal[1] - last_wp.y, self.goal[0] - last_wp.x)
                self.waypoints.append(Waypoint(
                    x=self.goal[0], y=self.goal[1],
                    yaw=goal_yaw, index=len(self.waypoints)
                ))
        
        self.rrt_path = path
        self.current_waypoint_idx = 0
        self.planning_success = True
        self.planning_time = self.rrt_planner.planning_time
        self.iterations_used = self.rrt_planner.iterations_used
        
        print(f"[MPC_RRT] Path planned: {len(self.waypoints)} waypoints, "
              f"time={self.planning_time:.3f}s, iterations={self.iterations_used}")
        
        return True
    
    def _get_current_target(self, current_state: List[float]) -> Tuple[float, float]:
        """
        Get the current target waypoint based on vehicle position.
        Uses lookahead to find appropriate target waypoint.
        
        Args:
            current_state: [x, y, v, psi] current state
            
        Returns:
            (x, y) target position
        """
        if not self.waypoints:
            return self.goal
        
        x, y = current_state[0], current_state[1]
        v = current_state[2]
        
        # Update waypoint index based on progress
        while self.current_waypoint_idx < len(self.waypoints) - 1:
            wp = self.waypoints[self.current_waypoint_idx]
            dist_to_wp = math.hypot(x - wp.x, y - wp.y)
            
            if dist_to_wp < self.waypoint_tolerance:
                self.current_waypoint_idx += 1
            else:
                break
        
        # Find lookahead waypoint (based on velocity)
        lookahead = max(self.lookahead_distance, v * 2.0)  # At least 2 seconds ahead
        
        best_wp_idx = self.current_waypoint_idx
        accumulated_dist = 0
        
        for i in range(self.current_waypoint_idx, len(self.waypoints) - 1):
            wp_current = self.waypoints[i]
            wp_next = self.waypoints[i + 1]
            segment_dist = math.hypot(wp_next.x - wp_current.x, wp_next.y - wp_current.y)
            accumulated_dist += segment_dist
            
            if accumulated_dist >= lookahead:
                best_wp_idx = i + 1
                break
            best_wp_idx = i + 1
        
        # Ensure we don't go beyond waypoints
        best_wp_idx = min(best_wp_idx, len(self.waypoints) - 1)
        
        target_wp = self.waypoints[best_wp_idx]
        return (target_wp.x, target_wp.y)
    
    def _check_deviation(self, current_state: List[float]) -> float:
        """
        Check deviation from planned path.
        
        Args:
            current_state: [x, y, v, psi] current state
            
        Returns:
            Minimum distance to path
        """
        if not self.rrt_path or len(self.rrt_path) < 2:
            return 0.0
        
        x, y = current_state[0], current_state[1]
        min_dist = float('inf')
        
        # Check distance to path segments
        for i in range(len(self.rrt_path) - 1):
            p1 = self.rrt_path[i]
            p2 = self.rrt_path[i + 1]
            
            # Point-to-segment distance
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            seg_len_sq = dx * dx + dy * dy
            
            if seg_len_sq < 1e-6:
                dist = math.hypot(x - p1.x, y - p1.y)
            else:
                t = max(0, min(1, ((x - p1.x) * dx + (y - p1.y) * dy) / seg_len_sq))
                proj_x = p1.x + t * dx
                proj_y = p1.y + t * dy
                dist = math.hypot(x - proj_x, y - proj_y)
            
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def update_dynamic_obstacles(self, dynamic_obstacles: List[Dict]):
        """
        Update dynamic obstacles for MPC.
        
        Args:
            dynamic_obstacles: New list of dynamic obstacles
        """
        self.dynamic_obstacles = dynamic_obstacles
        # Note: MPC needs to be recreated to update dynamic obstacles
        # This is because acados compiles the solver with fixed obstacle count
    
    def solve(self, cur_state: List[float], debug: bool = False) -> Tuple[np.ndarray, int, Dict]:
        """
        Solve the integrated RRT+MPC control problem using pre-compiled solver pool.
        
        1. Select appropriate pre-compiled solver based on progress
        2. Update waypoint progress tracking  
        3. Solve MPC for waypoint tracking
        
        Args:
            cur_state: [x, y, v, psi] current state
            debug: Enable debug output
            
        Returns:
            (control, solver_status, debug_info)
            - control: np.array([steering, acceleration])
            - solver_status: 0 if optimal, non-zero otherwise
            - debug_info: Dictionary with debug information
        """
        x, y, v, psi = cur_state
        
        # Check if we need to replan (large deviation from path)
        deviation = self._check_deviation(cur_state)
        if deviation > self.replan_on_deviation:
            print(f"[MPC_RRT] Large deviation ({deviation:.2f}m), replanning...")
            self._plan_rrt_path(cur_state)
            self._precompile_solvers()  # Re-compile all solvers for new path
            self.current_solver_idx = 0
        
        # Select appropriate pre-compiled solver based on progress
        new_solver_idx = self._select_solver(cur_state)
        if new_solver_idx != self.current_solver_idx:
            self.current_solver_idx = new_solver_idx
            self.mpc = self.solver_pool[new_solver_idx]
            self.current_mpc_target = self.solver_targets[new_solver_idx]
            # print(f"[MPC_RRT] Switched to solver {new_solver_idx}, target={self.current_mpc_target}")
        
        # KEY FIX: Update the shared MPC solver's goal to the current target
        # Since we use a single solver instance, we must update the goal dynamically
        # This fixes the issue where the solver was stuck targeting the final goal
        # or the previous target.
        # MPC_core.py has been updated to accept dynamic goal via parameters 'p'
        # but it reads self.goal in solve() to populate 'p'.
        if hasattr(self, 'solver_targets') and self.current_solver_idx < len(self.solver_targets):
            self.mpc.goal = self.solver_targets[self.current_solver_idx]
            
        # Update waypoint progress tracking (for visualization)
        self._update_waypoint_progress(cur_state)
        
        # Get current target waypoint (for visualization/debug)
        target = self._get_current_target(cur_state)
        
        # Check if we've reached the final goal
        goal_dist = math.hypot(x - self.goal[0], y - self.goal[1])
        if goal_dist < 0.5:
            # Very close to goal - slow down and stop
            self.debug_info = {
                'target_waypoint': target,
                'current_waypoint_idx': self.current_waypoint_idx,
                'total_waypoints': len(self.waypoints),
                'deviation': deviation,
                'goal_reached': True,
                'solver_idx': self.current_solver_idx
            }
            return np.array([0.0, -1.0]), 0, self.debug_info
        
        # Check minimum distance to obstacles
        min_obs_dist = self._get_min_obstacle_distance(x, y)
        
        # ALWAYS use MPC - Pure Pursuit doesn't avoid obstacles
        # MPC with pre-compiled solver pool provides both path following and obstacle avoidance
        use_pure_pursuit = False  # Disabled - MPC handles everything
        
        if use_pure_pursuit:
            # Pure Pursuit controller for path following
            control = self._pure_pursuit_control(cur_state, target)
            status = 0
            mpc_debug = {'mode': 'pure_pursuit', 'min_obs_dist': min_obs_dist}
        else:
            # Get RRT-based avoidance hint from upcoming waypoints
            avoidance_hint = self._get_avoidance_hint(cur_state)
            
            # Let MPC use its own obstacle-aware initialization
            # The pre-compiled solver targets the correct waypoint
            control, status, mpc_debug = self.mpc.solve(
                np.array(cur_state), 
                debug=debug, 
                skip_initialization=False,  # Let MPC handle initialization
                avoidance_hint=avoidance_hint
            )
            mpc_debug = mpc_debug or {}
            mpc_debug['mode'] = 'mpc'
            mpc_debug['min_obs_dist'] = min_obs_dist
            mpc_debug['avoidance_hint'] = avoidance_hint
        
        # Build debug info
        self.debug_info = {
            'target_waypoint': target,
            'mpc_target': self.current_mpc_target,
            'current_waypoint_idx': self.current_waypoint_idx,
            'target_waypoint_idx': self.target_waypoint_idx,
            'total_waypoints': len(self.waypoints),
            'deviation': deviation,
            'goal_reached': False,
            'mpc_status': status,
            'goal_distance': goal_dist,
            'solver_idx': self.current_solver_idx,
            'num_solvers': len(self.solver_pool)
        }
        
        if debug:
            self.debug_info['mpc_debug'] = mpc_debug
            self.debug_info['all_waypoints'] = [(wp.x, wp.y) for wp in self.waypoints]
        
        return control, status, self.debug_info
    
    def _update_waypoint_progress(self, cur_state: List[float]):
        """
        Update waypoint progress tracking (for visualization only).
        NO MPC recompilation - we use pre-compiled solver pool.
        
        Args:
            cur_state: [x, y, v, psi] current state
        """
        if not self.waypoints:
            return
        
        x, y = cur_state[0], cur_state[1]
        
        # Update current waypoint index based on proximity
        waypoint_reach_threshold = 2.0  # meters
        while self.current_waypoint_idx < len(self.waypoints) - 1:
            wp = self.waypoints[self.current_waypoint_idx]
            dist_to_wp = math.hypot(x - wp.x, y - wp.y)
            if dist_to_wp < waypoint_reach_threshold:
                self.current_waypoint_idx += 1
            else:
                break
    
    def _update_waypoint_target(self, cur_state: List[float], force: bool = False):
        """
        DEPRECATED: No longer rebuilds MPC at runtime.
        Use pre-compiled solver pool instead via _select_solver().
        Kept for backward compatibility.
        """
        self._update_waypoint_progress(cur_state)
    
    def _update_mpc_target(self, cur_state: List[float], force_update: bool = False):
        """
        Update MPC target to follow waypoints progressively.
        Recreates MPC when target changes significantly.
        
        Args:
            cur_state: [x, y, v, psi] current state
            force_update: If True, force target update regardless of distance
        """
        if not self.waypoints:
            return
            
        x, y = cur_state[0], cur_state[1]
        
        # Check distance to current MPC target
        dist_to_target = math.hypot(x - self.current_mpc_target[0], y - self.current_mpc_target[1])
        
        # If we're very close to current target, update to next target
        # Use a larger threshold to avoid frequent MPC recompilation
        target_reached_threshold = 8.0  # meters - only update when quite close
        
        if dist_to_target < target_reached_threshold or force_update:
            # Find next target: look far ahead on the path
            lookahead_count = 8  # Look 8 waypoints ahead for smoother path
            
            # Find closest waypoint to current position
            closest_idx = 0
            closest_dist = float('inf')
            for i, wp in enumerate(self.waypoints):
                d = math.hypot(x - wp.x, y - wp.y)
                if d < closest_dist:
                    closest_dist = d
                    closest_idx = i
            
            # Target is lookahead_count waypoints ahead
            target_idx = min(closest_idx + lookahead_count, len(self.waypoints) - 1)
            
            # Last few waypoints: target the final goal
            if target_idx >= len(self.waypoints) - 3:
                new_target = self.goal
            else:
                new_target = (self.waypoints[target_idx].x, self.waypoints[target_idx].y)
            
            # Check if target changed significantly - require large change to recompile
            target_change = math.hypot(new_target[0] - self.current_mpc_target[0],
                                       new_target[1] - self.current_mpc_target[1])
            
            if target_change > 5.0 or force_update:
                # Recreate MPC with new target
                self.current_mpc_target = new_target
                
                # Suppress recompilation messages
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    self.mpc = MPC(
                        static_obstacles=self.static_obstacles,
                        dynamic_obstacles=self.dynamic_obstacles,
                        horizon=self.horizon,
                        dt=self.dt,
                        goal=new_target
                    )
                finally:
                    sys.stdout = old_stdout
    
    def _update_mpc_goal(self, new_goal: Tuple[float, float]):
        """
        Update the MPC goal for waypoint tracking.
        
        Note: acados does not support updating yref at runtime for NONLINEAR_LS cost.
        The residual formulation uses symbolic goal, so we just update the internal goal.
        The cost function residuals are computed relative to self.mpc.goal in the solver.
        
        For true online goal updates, we would need to use parameters (p) instead of
        hardcoded goal in the cost function. For now, we only recreate MPC when the
        goal changes significantly (> threshold distance).
        
        Args:
            new_goal: (x, y) new target position
        """
        import casadi as ca
        
        # Check if goal change is significant enough to warrant recreation
        current_goal = (float(self.mpc.goal[0]), float(self.mpc.goal[1]))
        dist = math.hypot(new_goal[0] - current_goal[0], new_goal[1] - current_goal[1])
        
        # Only recreate if goal moved more than 5 meters (avoid constant recompilation)
        if dist > 5.0:
            # Update internal goal for cost function (this affects warm-start only)
            self.mpc.goal = ca.DM([new_goal[0], new_goal[1]])
            # Note: The actual residual computation in acados uses the compiled goal,
            # but the solve() method uses warm-start initialization with self.mpc.goal
            # This is sufficient for waypoint tracking as long as MPC follows the path
    
    def _initialize_mpc_with_rrt_path(self, cur_state: List[float]):
        """
        Initialize MPC trajectory using RRT waypoints.
        This helps MPC avoid local minima by providing a good initial guess.
        
        Args:
            cur_state: [x, y, v, psi] current state
        """
        if not self.waypoints or self.current_waypoint_idx >= len(self.waypoints):
            return  # No waypoints, let MPC use default initialization
        
        x0, y0, v0, psi0 = cur_state
        v_init = max(v0, 2.0)  # Assume reasonable speed
        
        # Find waypoints ahead of current position
        remaining_waypoints = self.waypoints[self.current_waypoint_idx:]
        
        # Interpolate RRT path to match MPC horizon
        path_points = [(x0, y0)]  # Start with current position
        for wp in remaining_waypoints:
            path_points.append((wp.x, wp.y))
        
        # Calculate cumulative distances along path
        cum_dist = [0.0]
        for i in range(1, len(path_points)):
            d = math.hypot(path_points[i][0] - path_points[i-1][0],
                          path_points[i][1] - path_points[i-1][1])
            cum_dist.append(cum_dist[-1] + d)
        
        total_dist = cum_dist[-1]
        if total_dist < 0.1:
            return  # Path too short
        
        # Initialize each horizon step
        for i in range(self.horizon + 1):
            t = i * self.dt
            target_dist = min(v_init * t, total_dist)
            
            # Find the segment containing target_dist
            seg_idx = 0
            for j in range(len(cum_dist) - 1):
                if cum_dist[j] <= target_dist <= cum_dist[j + 1]:
                    seg_idx = j
                    break
                seg_idx = j
            
            # Interpolate within segment
            if seg_idx < len(path_points) - 1:
                seg_start = cum_dist[seg_idx]
                seg_end = cum_dist[seg_idx + 1]
                seg_len = seg_end - seg_start
                
                if seg_len > 0.01:
                    alpha = (target_dist - seg_start) / seg_len
                else:
                    alpha = 0.0
                alpha = min(1.0, max(0.0, alpha))
                
                x_init = path_points[seg_idx][0] + alpha * (path_points[seg_idx + 1][0] - path_points[seg_idx][0])
                y_init = path_points[seg_idx][1] + alpha * (path_points[seg_idx + 1][1] - path_points[seg_idx][1])
                
                # Compute heading towards next point
                if seg_idx < len(path_points) - 1:
                    psi_init = math.atan2(path_points[seg_idx + 1][1] - path_points[seg_idx][1],
                                          path_points[seg_idx + 1][0] - path_points[seg_idx][0])
                else:
                    psi_init = psi0
            else:
                # Beyond path, use last waypoint
                x_init = path_points[-1][0]
                y_init = path_points[-1][1]
                psi_init = math.atan2(self.goal[1] - y_init, self.goal[0] - x_init)
            
            # Set initial guess for this horizon step
            try:
                self.mpc.solver.set(i, 'x', np.array([x_init, y_init, v_init, psi_init]))
            except:
                pass  # If setting fails, use default
        
        # Initialize controls
        for i in range(self.horizon):
            try:
                self.mpc.solver.set(i, 'u', np.array([0.0, 0.5]))  # Gentle acceleration
            except:
                pass
    
    def _get_avoidance_hint(self, cur_state: List[float]) -> Optional[Tuple[float, float]]:
        """
        Get an avoidance hint point from the RRT path.
        This helps MPC know which direction to go around obstacles.
        
        Args:
            cur_state: [x, y, v, psi] current state
            
        Returns:
            (x, y) point that indicates the desired avoidance direction, or None
        """
        if not self.waypoints or len(self.waypoints) < 2:
            return None
        
        x0, y0 = cur_state[0], cur_state[1]
        v = cur_state[2]
        
        # First, find the closest waypoint to current position
        min_dist = float('inf')
        closest_idx = 0
        for i, wp in enumerate(self.waypoints):
            d = math.hypot(wp.x - x0, wp.y - y0)
            if d < min_dist:
                min_dist = d
                closest_idx = i
        
        # Look ahead on the RRT path from the closest waypoint
        lookahead_dist = max(5.0, v * 2.5)  # At least 5m or 2.5 seconds ahead
        
        # Start from the waypoint AFTER the closest one
        start_idx = min(closest_idx + 1, len(self.waypoints) - 1)
        
        # Accumulate distance along path to find hint point
        accumulated_dist = math.hypot(self.waypoints[start_idx].x - x0, 
                                       self.waypoints[start_idx].y - y0)
        hint_point = (self.waypoints[start_idx].x, self.waypoints[start_idx].y)
        
        for i in range(start_idx, len(self.waypoints)):
            wp = self.waypoints[i]
            if i > start_idx:
                prev_wp = self.waypoints[i - 1]
                accumulated_dist += math.hypot(wp.x - prev_wp.x, wp.y - prev_wp.y)
            
            hint_point = (wp.x, wp.y)
            
            if accumulated_dist >= lookahead_dist:
                break
        
        return hint_point
    
    def _get_min_obstacle_distance(self, x: float, y: float) -> float:
        """
        Get minimum distance to any obstacle.
        
        Args:
            x, y: Current position
            
        Returns:
            Minimum distance to obstacle surface (accounting for vehicle radius)
        """
        min_dist = float('inf')
        vehicle_radius = cfg.vehicle_radius
        
        for obs in self.static_obstacles:
            pos = obs['position']
            r = obs['radius']
            dist = math.hypot(x - pos[0], y - pos[1]) - r - vehicle_radius
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _pure_pursuit_control(self, cur_state: List[float], target: Tuple[float, float]) -> np.ndarray:
        """
        Simple Pure Pursuit controller for path following.
        
        Args:
            cur_state: [x, y, v, psi] current state
            target: (x, y) target position
            
        Returns:
            control: np.array([steering, acceleration])
        """
        x, y, v, psi = cur_state
        tx, ty = target
        
        # Vehicle parameters
        L = cfg.vehicle_length  # wheelbase
        
        # Calculate target angle
        dx = tx - x
        dy = ty - y
        target_dist = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        
        # Heading error
        heading_error = target_angle - psi
        # Normalize to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Pure Pursuit: steering = atan(2 * L * sin(alpha) / ld)
        # where alpha is heading error and ld is lookahead distance
        ld = max(target_dist, 2.0)  # Lookahead distance, at least 2m
        
        # Curvature
        if ld > 0.1:
            curvature = 2.0 * math.sin(heading_error) / ld
        else:
            curvature = 0.0
        
        # Steering angle
        steering = math.atan(L * curvature)
        steering = max(-cfg.max_steering_angle, min(cfg.max_steering_angle, steering))
        
        # Speed control - slow down for tight turns
        turn_factor = 1.0 - 0.5 * abs(steering) / cfg.max_steering_angle
        target_speed = cfg.max_speed * turn_factor * 0.7  # 70% of max for safety
        
        # Acceleration
        if v < target_speed:
            accel = min(cfg.max_acceleration, (target_speed - v) * 2.0)
        else:
            accel = max(cfg.max_deceleration, (target_speed - v) * 2.0)
        
        return np.array([steering, accel])
    
    def get_path_visualization(self) -> Dict[str, List]:
        """
        Get path data for visualization.
        
        Returns:
            Dictionary with 'rrt_path' and 'waypoints' for plotting
        """
        result = {
            'rrt_path_x': [],
            'rrt_path_y': [],
            'waypoint_x': [],
            'waypoint_y': [],
            'current_target': None
        }
        
        if self.rrt_path:
            result['rrt_path_x'] = [s.x for s in self.rrt_path]
            result['rrt_path_y'] = [s.y for s in self.rrt_path]
        
        if self.waypoints:
            result['waypoint_x'] = [wp.x for wp in self.waypoints]
            result['waypoint_y'] = [wp.y for wp in self.waypoints]
            
            if self.current_waypoint_idx < len(self.waypoints):
                wp = self.waypoints[self.current_waypoint_idx]
                result['current_target'] = (wp.x, wp.y)
        
        return result
    
    def get_planning_stats(self) -> Dict[str, Any]:
        """
        Get planning statistics.
        
        Returns:
            Dictionary with planning statistics
        """
        return {
            'planning_success': self.planning_success,
            'planning_time': self.planning_time,
            'iterations_used': self.iterations_used,
            'num_waypoints': len(self.waypoints),
            'path_length': sum(
                math.hypot(self.waypoints[i+1].x - self.waypoints[i].x,
                          self.waypoints[i+1].y - self.waypoints[i].y)
                for i in range(len(self.waypoints) - 1)
            ) if len(self.waypoints) > 1 else 0
        }


# ============== Convenience Functions ==============

def create_mpc_rrt_controller(start_state: List[float], goal: Tuple[float, float],
                               static_obstacles: List[Dict], **kwargs) -> MPC_RRT_Controller:
    """
    Factory function to create MPC_RRT controller with default parameters.
    
    Args:
        start_state: [x, y, v, psi] initial state
        goal: (x, y) goal position
        static_obstacles: List of static obstacles
        **kwargs: Additional parameters for MPC_RRT_Controller
        
    Returns:
        Configured MPC_RRT_Controller instance
    """
    default_params = {
        'map_size': 45.0,  # Must be larger than goal position
        'horizon': 40,
        'dt': 0.2,
        'waypoint_tolerance': 2.0,
        'lookahead_distance': 4.0,
        'replan_on_deviation': 8.0
    }
    default_params.update(kwargs)
    
    return MPC_RRT_Controller(
        start_state=start_state,
        goal=goal,
        static_obstacles=static_obstacles,
        **default_params
    )


if __name__ == "__main__":
    """Test the MPC_RRT controller."""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared_config import STATIC_OBSTACLES, START_POS, GOAL_POS, START_YAW
    
    # Convert obstacles to MPC format
    obstacles = []
    for obs in STATIC_OBSTACLES:
        obstacles.append({
            'position': obs['position'],
            'type': 'circle',
            'radius': obs['radius']
        })
    
    # Initial state
    start_state = [START_POS[0], START_POS[1], 0.5, START_YAW]
    
    print("=" * 60)
    print("Testing MPC_RRT Controller")
    print("=" * 60)
    
    # Create controller
    controller = create_mpc_rrt_controller(
        start_state=start_state,
        goal=GOAL_POS,
        static_obstacles=obstacles
    )
    
    # Print planning stats
    stats = controller.get_planning_stats()
    print(f"\nPlanning Statistics:")
    print(f"  Success: {stats['planning_success']}")
    print(f"  Time: {stats['planning_time']:.3f}s")
    print(f"  Iterations: {stats['iterations_used']}")
    print(f"  Waypoints: {stats['num_waypoints']}")
    print(f"  Path Length: {stats['path_length']:.2f}m")
    
    # Print waypoints
    print(f"\nWaypoints:")
    for i, wp in enumerate(controller.waypoints):
        print(f"  {i}: ({wp.x:.2f}, {wp.y:.2f})")
    
    # Test solve
    print(f"\nTesting solve()...")
    control, status, info = controller.solve(start_state, debug=True)
    print(f"  Control: steering={control[0]:.3f}, accel={control[1]:.3f}")
    print(f"  Status: {status}")
    print(f"  Target: {info['target_waypoint']}")
    print(f"  Waypoint: {info['current_waypoint_idx']}/{info['total_waypoints']}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
