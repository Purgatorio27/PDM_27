import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from Obstacles import distance
import Config as cfg
import numpy as np
import math

class MPC:
    """
    Nonlinear Model Predictive Control (NMPC) module to compute optimal input sequence at a given step.
    """
    def __init__(self, static_obstacles, dynamic_obstacles, horizon = 20, dt = 0.1, goal=None, compile=True):
        self.solver = None
        self.ocp = None
        self.compile_solver = compile

        self.nx = 4       # number of states
        self.nu = 2       # number of inputs

        # self.car_axis_param = car_axis_param # l_f and l_r of the vehicle
        # self.goal = goal    # goal position

        self.N = horizon    # prediction horizon
        self.dt = dt        # time step
        self.cfg = cfg      # configuration file
        # Use provided goal or default from config
        self.goal = ca.DM(goal) if goal is not None else ca.DM(self.cfg.goal)
        self.l_f = self.cfg.lf
        self.l_r = self.cfg.lr
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles  # Dynamic obstacles CURRENT positions and velocities (one time step)

        self._create_ocp()
    

    def _create_ocp(self):
        """
        Create the optimal control problem (OCP) for the vehicle MPC.
        """

        self.ocp = AcadosOcp()
        self.ocp.dims.N = self.N
        self.ocp.dims.nx = self.nx  # = 4
        self.ocp.dims.nu = self.nu  # = 2

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        v = ca.SX.sym('v')

        psi = ca.SX.sym('psi')  # yaw angle
        self.states = ca.vertcat(x, y, v, psi) # state vector

        delta = ca.SX.sym('delta')
        a = ca.SX.sym('a')
        self.controls = ca.vertcat(delta, a)  # steering angle and acceleration

        dt = self.dt
        l_f = self.l_f
        l_r = self.l_r
        beta = ca.atan((l_r / (l_f + l_r)) * ca.tan(delta)) # slip angle
        
        # vehicle dynamics - first order Euler discretization
        x_next = x + v * ca.cos(psi + beta) * dt
        y_next = y + v * ca.sin(psi + beta) * dt
        v_next = v + a * dt
        psi_next = psi + (v / l_r) * ca.sin(beta) * dt  
        f_discrete = ca.vertcat(x_next, y_next, v_next, psi_next)
        
        # Extra parameters 
        self.num_static_obs = len(self.static_obstacles)
        self.num_dyn_obs = len(self.dynamic_obstacles)

        # Extend parameters to include dynamic goal (x, y)
        num_params = self.num_dyn_obs * 2 + 2 
        p = ca.SX.sym('p', num_params)
        
        # Extract goal parameters
        self.p_goal_x = p[num_params-2]
        self.p_goal_y = p[num_params-1]
        
        # Update self.goal to be symbolic representation for cost definition
        # We assume self.goal was initialized with DM values, we overwrite it with symbolic
        # But we need to keep the numeric value for initial guesses? 
        # self.goal is used in add_costs via self.goal[0] etc.
        # We can just change self.goal to be a list of symbolic expressions
        self.goal_numeric = self.goal # Backup numeric goal
        self.goal = ca.vertcat(self.p_goal_x, self.p_goal_y)

        self.ocp.model.name = "vehicle_mpc"
        self.ocp.model.x = self.states
        self.ocp.model.u = self.controls
        self.ocp.model.disc_dyn_expr = f_discrete
        self.ocp.model.p = p
        self.ocp.parameter_values = np.zeros((num_params,))


        self.setup_hard_constraints()
        self.add_costs()

        # Solver options
        self.ocp.solver_options.tf = self.N * self.dt
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'DISCRETE'
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # RTI for real-time, more robust

        self.ocp.solver_options.qp_solver_iter_max = 200
        self.ocp.solver_options.nlp_solver_max_iter = 100  # Fewer iterations for stability
        self.ocp.solver_options.tol = 1e-3  # Relaxed tolerance for stability
        self.ocp.solver_options.qp_solver_cond_N = 5  # partial condensing
        
        # Regularization for numerical stability with EXTERNAL cost
        self.ocp.solver_options.levenberg_marquardt = 1.0  # Stronger regularization
        
        self.solver = AcadosOcpSolver(
            self.ocp, 
            json_file='vehicle_mpc.json',
            build=self.compile_solver,
            generate=self.compile_solver
        )


    def setup_hard_constraints(self):   
        """
        Define constraints for the MPC problem.
        Obstacle avoidance is handled ONLY by cost function for better performance.
        """
        
        self.ocp.constraints.x0 = ca.DM.zeros(self.nx)  # initial state constraint placeholder
        self.ocp.constraints.lbx = [self.cfg.min_speed]
        self.ocp.constraints.ubx = [self.cfg.max_speed]
        self.ocp.constraints.idxbx = [2]  # speed index

        self.ocp.constraints.lbu = [-self.cfg.max_steering_angle, self.cfg.max_deceleration]
        self.ocp.constraints.ubu = [self.cfg.max_steering_angle, self.cfg.max_acceleration]
        self.ocp.constraints.idxbu = [0, 1]  # steering angle and acceleration indices
        
        # NO obstacle constraints - handled purely by cost function 
        # This prevents the solver from being overly conservative


    def get_predicted_obs_trajectories(self, i): # i is the time step within horizon
        """
        Retrieve predicted trajectories of dynamic obstacles over the horizon.
        """
        predicted_obstacle_trajectories = []

        for obs in self.dynamic_obstacles:
            x, y, vx, vy = obs['trajectory']
            x_pred = x + vx * i * self.dt
            y_pred = y + vy * i * self.dt
            predicted_obstacle_trajectories.append(x_pred)
            predicted_obstacle_trajectories.append(y_pred)

        return np.array(predicted_obstacle_trajectories)
            

    def add_costs(self):
        """
        Define the cost function for the MPC problem.
        Uses NONLINEAR_LS (residual form) for compatibility with GAUSS_NEWTON Hessian.
        
        The cost is: 0.5 * ||y_model - y_ref||^2_W
        We construct residual vector y_model and set W (weights) and y_ref (reference).
        """
        # Weights for different cost components
        W_goal_x = 5.0       # Goal position weight
        W_goal_y = 5.0
        W_heading = 2.0      # Heading alignment weight
        W_velocity = 2.0     # Velocity tracking weight
        W_steer = 0.01       # Steering effort
        W_accel = 0.01      # Acceleration effort
        W_obs = 10000.0       # Obstacle avoidance weight (Increased significantly for safety)
        W_obs_velocity = 500.0  # Velocity-dependent obstacle penalty (Increased)
        
        # Safety margin for obstacle avoidance
        # distance() returns SURFACE-TO-SURFACE distance (clearance).
        # A margin of 0.5m is sufficient for safety.
        # Previously set to 3.0 which forced vehicle 3 meters away from everything.
        safety_margin = 0.8  # meters clearance


        
        # Build residual vector y_model
        residuals = []
        weights = []
        refs = []
        
        # 1. Goal position residuals (x - goal_x, y - goal_y)
        residuals.append(self.states[0] - self.goal[0])
        residuals.append(self.states[1] - self.goal[1])
        weights.extend([W_goal_x, W_goal_y])
        refs.extend([0.0, 0.0])
        
        # 2. Heading alignment residual - penalize deviation from goal direction
        goal_angle = ca.atan2(self.goal[1] - self.states[1], self.goal[0] - self.states[0])
        heading_error = ca.atan2(ca.sin(goal_angle - self.states[3]), ca.cos(goal_angle - self.states[3]))
        residuals.append(heading_error)
        weights.append(W_heading)
        refs.append(0.0)
        
        # 3. Velocity tracking residual - target moderate speed for navigation
        v_ref = 0.35 * cfg.max_speed  # Target speed ~1.4 m/s
        residuals.append(self.states[2] - v_ref)
        weights.append(W_velocity)
        refs.append(0.0)
        
        # 4. Control effort residuals
        residuals.append(self.controls[0])  # steering
        residuals.append(self.controls[1])  # acceleration
        weights.extend([W_steer, W_accel])
        refs.extend([0.0, 0.0])
        
        # 5. Static obstacle avoidance residuals
        # Two types of residuals:
        # a) Position-based: penalize being close to obstacle
        # b) Velocity-based: penalize high speed when near obstacle
        alpha = 3.0  # Smoothness parameter for softplus
        
        for static_obs in self.static_obstacles:
            position = ca.DM(static_obs['position'])
            radius = static_obs['radius']
            dist = distance(self.states[0:2], position, radius1=self.cfg.vehicle_radius, radius2=radius)
            
            # a) Position-based violation
            margin_violation = safety_margin - dist
            smooth_violation = (1.0/alpha) * ca.log(1.0 + ca.exp(alpha * margin_violation))
            residuals.append(smooth_violation)
            weights.append(W_obs)
            refs.append(0.0)
            
            # b) Velocity-dependent violation: penalize high speed when near obstacle
            # When close to obstacle, we want v → 0, so residual = v * proximity_factor
            # proximity_factor = softplus(close_margin - dist) where close_margin is smaller
            close_margin = 1.5  # Start velocity penalty at 1.5m clearance
            proximity_violation = close_margin - dist
            proximity_factor = (1.0/alpha) * ca.log(1.0 + ca.exp(alpha * proximity_violation))
            # Residual = v * proximity_factor (when close and fast, high penalty)
            velocity_obs_residual = self.states[2] * proximity_factor
            residuals.append(velocity_obs_residual)
            weights.append(W_obs_velocity)
            refs.append(0.0)
        
        # 6. Dynamic obstacle avoidance residuals
        for i in range(self.num_dyn_obs):
            obs_x = self.ocp.model.p[i * 2]
            obs_y = self.ocp.model.p[i * 2 + 1]
            dist_dyn = distance(self.states[0:2], ca.vertcat(obs_x, obs_y),
                                radius1=self.cfg.vehicle_radius, radius2=self.dynamic_obstacles[i]['radius'])
            
            # Position-based violation
            margin_violation_dyn = safety_margin - dist_dyn
            smooth_violation_dyn = (1.0/alpha) * ca.log(1.0 + ca.exp(alpha * margin_violation_dyn))
            residuals.append(smooth_violation_dyn)
            weights.append(W_obs)
            refs.append(0.0)
            
            # Velocity-dependent violation
            proximity_violation_dyn = close_margin - dist_dyn
            proximity_factor_dyn = (1.0/alpha) * ca.log(1.0 + ca.exp(alpha * proximity_violation_dyn))
            velocity_obs_residual_dyn = self.states[2] * proximity_factor_dyn
            residuals.append(velocity_obs_residual_dyn)
            weights.append(W_obs_velocity)
            refs.append(0.0)
        
        # Construct residual vector and weight matrix
        ny = len(residuals)
        y_model = ca.vertcat(*residuals)
        W = np.diag(weights)
        y_ref = np.array(refs)
        
        # Set NONLINEAR_LS cost
        self.ocp.model.cost_y_expr = y_model
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.cost.W = W
        self.ocp.cost.yref = y_ref
        self.ocp.dims.ny = ny
        
        # Terminal cost - simplified goal attraction
        # Terminal residuals (no control inputs, stronger goal weight)
        terminal_residuals = []
        terminal_weights = []
        terminal_refs = []
        
        # Goal position (stronger weight at terminal)
        terminal_residuals.append(self.states[0] - self.goal[0])
        terminal_residuals.append(self.states[1] - self.goal[1])
        terminal_weights.extend([8.0, 8.0])  # Stronger terminal goal weight
        terminal_refs.extend([0.0, 0.0])
        
        # Terminal heading
        terminal_residuals.append(heading_error)
        terminal_weights.append(2.0)
        terminal_refs.append(0.0)
        
        # # Terminal obstacle avoidance (position + velocity)
        for static_obs in self.static_obstacles:
            position = ca.DM(static_obs['position'])
            radius = static_obs['radius']
            dist = distance(self.states[0:2], position, radius1=self.cfg.vehicle_radius, radius2=radius)
            
            # Position-based
            margin_violation = safety_margin - dist
            smooth_violation = (1.0/alpha) * ca.log(1.0 + ca.exp(alpha * margin_violation))
            terminal_residuals.append(smooth_violation)
            terminal_weights.append(W_obs * 1.5)  # Even stronger at terminal
            terminal_refs.append(0.0)
            
            # Velocity-based
            proximity_violation = close_margin - dist
            proximity_factor = (1.0/alpha) * ca.log(1.0 + ca.exp(alpha * proximity_violation))
            velocity_obs_residual = self.states[2] * proximity_factor
            terminal_residuals.append(velocity_obs_residual)
            terminal_weights.append(W_obs_velocity * 1.5)
            terminal_refs.append(0.0)
        
        ny_e = len(terminal_residuals)
        y_model_e = ca.vertcat(*terminal_residuals)
        W_e = np.diag(terminal_weights)
        y_ref_e = np.array(terminal_refs)
        
        self.ocp.model.cost_y_expr_e = y_model_e
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.cost.W_e = W_e
        self.ocp.cost.yref_e = y_ref_e
        self.ocp.dims.ny_e = ny_e


    def solve(self, cur_state, debug=False, skip_initialization=False, avoidance_hint=None):
        """
        Solve the MPC problem for the current state and dynamic obstacle positions.
        
        Args:
            cur_state: Current state [x, y, v, psi]
            debug: Enable debug output
            skip_initialization: If True, skip trajectory initialization (use external init)
            avoidance_hint: Optional (x, y) point that indicates the desired avoidance direction.
                           MPC will try to go through this point when avoiding obstacles.
        
        Returns: (control, solver_status, debug_info)
        """
        # Set initial state constraint
        self.solver.set(0, 'lbx', cur_state) # cur_state: [x, y, v, psi] numpy array
        self.solver.set(0, 'ubx', cur_state)
        
        x0, y0, v0, psi0 = cur_state
        
        # Handle symbolic vs numeric goal
        if isinstance(self.goal, (ca.SX, ca.MX)):
             if hasattr(self, 'goal_numeric'):
                 goal_x = float(self.goal_numeric[0])
                 goal_y = float(self.goal_numeric[1])
             else:
                 goal_x, goal_y = 0.0, 0.0 # Should not happen
        else:
             goal_x = float(self.goal[0])
             goal_y = float(self.goal[1])
        
        if not skip_initialization:
            # Initialize trajectory with obstacle-aware path
            # Compute direction to goal
            goal_angle = math.atan2(goal_y - y0, goal_x - x0)
            v_init = max(v0, 2.0)
            
            # Find ALL obstacles that block the straight path to goal
            blocking_obstacles = []
            for obs in self.static_obstacles:
                ox, oy = obs['position']
                r = obs['radius']

                # Check if obstacle intersects with path from current pos to goal
                # Using perpendicular distance from obstacle center to the line
                dx = goal_x - x0
                dy = goal_y - y0
                line_len = math.sqrt(dx**2 + dy**2)
                if line_len < 0.1:
                    continue

                # Perpendicular distance from obstacle to line
                cross = abs((ox - x0) * dy - (oy - y0) * dx) / line_len

                # Check if obstacle is between start and goal (not behind or beyond)
                dot = (ox - x0) * dx + (oy - y0) * dy
                if 0 < dot < line_len**2:
                    # Obstacle is along the path
                    safe_clearance = r + self.cfg.vehicle_radius + 2.5  # Extra margin
                    if cross < safe_clearance:
                        # Distance from start to obstacle projection on path
                        proj_dist = dot / line_len
                        blocking_obstacles.append({
                            'pos': (ox, oy),
                            'radius': r,
                            'proj_dist': proj_dist,
                            'perp_dist': cross
                        })

            # Sort by distance from start
            blocking_obstacles.sort(key=lambda x: x['proj_dist'])

            if blocking_obstacles:
                # Find the nearest blocking obstacle
                nearest_obs = blocking_obstacles[0]
                ox, oy = nearest_obs['pos']

                # Determine which side to go around (left or right of obstacle)
                if avoidance_hint is not None:
                    # Use avoidance_hint to determine direction
                    # avoidance_hint is a point (from RRT path) that indicates where to go
                    hint_x, hint_y = avoidance_hint
                    avoidance_angle = math.atan2(hint_y - y0, hint_x - x0)
                else:
                    # Default: use cross product to determine side
                    dx_to_obs = ox - x0
                    dy_to_obs = oy - y0
                    dx_to_goal = goal_x - x0
                    dy_to_goal = goal_y - y0
                    cross = dx_to_obs * dy_to_goal - dy_to_obs * dx_to_goal

                    # Use 60 degree avoidance angle
                    avoidance_angle = goal_angle + (math.pi/3 if cross > 0 else -math.pi/3)

                # Initialize with curved path - longer avoidance before blending back
                for i in range(self.N + 1):
                    t = i * self.dt
                    # Keep avoidance longer (5 seconds) before blending
                    blend = min(1.0, max(0.0, (t - 2.0) / 5.0))
                    current_angle = avoidance_angle * (1 - blend) + goal_angle * blend
                    x_init = x0 + v_init * math.cos(current_angle) * t
                    y_init = y0 + v_init * math.sin(current_angle) * t
                    self.solver.set(i, 'x', np.array([x_init, y_init, v_init, current_angle]))

                # Initialize controls for avoidance
                for i in range(self.N):
                    steer_init = (avoidance_angle - psi0) * 0.15 if i < 15 else 0.0
                    steer_init = max(-self.cfg.max_steering_angle, min(self.cfg.max_steering_angle, steer_init))
                    accel_init = 0.0
                    self.solver.set(i, 'u', np.array([steer_init, accel_init]))
            else:
                # No collision - use straight line
                for i in range(self.N + 1):
                    t = i * self.dt
                    x_init = x0 + v_init * math.cos(goal_angle) * t
                    y_init = y0 + v_init * math.sin(goal_angle) * t
                    self.solver.set(i, 'x', np.array([x_init, y_init, v_init, goal_angle]))
                
                for i in range(self.N):
                    steer_init = 0.0
                    accel_init = 0.5 if v0 < 2.0 else 0.0
                    self.solver.set(i, 'u', np.array([steer_init, accel_init]))

        # Set dynamic obstacle positions AND goal as parameters in all horizon steps
        
        # Handle symbolic vs numeric goal for parameter value
        if isinstance(self.goal, (ca.SX, ca.MX)):
             if hasattr(self, 'goal_numeric'):
                 g_x = float(self.goal_numeric[0])
                 g_y = float(self.goal_numeric[1])
             else:
                 g_x, g_y = 0.0, 0.0
        else:
             g_x = float(self.goal[0])
             g_y = float(self.goal[1])

        for i in range(self.N + 1):
            # For terminal stage N, we typically use the same parameters as N-1 or predicted
            # But get_predicted_obs_trajectories(i) handles i up to N (it uses i * dt)
            obs_pos_i = self.get_predicted_obs_trajectories(i)
            # Append goal params
            p_i = np.concatenate([obs_pos_i, [g_x, g_y]]) # NEW: Append goal
            self.solver.set(i, 'p', p_i)

        solve_MPC_status = self.solver.solve()
        
        debug_info = {}
        if debug:
            # Get predicted trajectory for debugging
            pred_x, pred_y = [], []
            for i in range(self.N + 1):
                state_i = self.solver.get(i, 'x')
                pred_x.append(float(state_i[0]))
                pred_y.append(float(state_i[1]))
            debug_info['predicted_trajectory'] = {'x': pred_x, 'y': pred_y}
            
            # Compute costs at initial state for debugging
            x, y, v, psi = cur_state
            
            # Use goal_numeric if available, else try simple goal
            if hasattr(self, 'goal_numeric') and self.goal_numeric is not None:
                 if isinstance(self.goal_numeric, (list, tuple, np.ndarray)):
                     try:
                        goal_x = float(self.goal_numeric[0])
                        goal_y = float(self.goal_numeric[1])
                     except:
                        goal_x, goal_y = 0.0, 0.0
                 else:
                     goal_x, goal_y = float(self.goal_numeric[0]), float(self.goal_numeric[1])
            else:
                 # Fallback if self.goal is still numeric (e.g. initial setup) or we risk crash
                 try:
                    goal_x, goal_y = float(self.goal[0]), float(self.goal[1])
                 except:
                    goal_x, goal_y = 0.0, 0.0

            # Goal direction
            goal_angle = math.atan2(goal_y - y, goal_x - x)
            heading_error = goal_angle - psi
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            
            # Distance to goal
            dist_to_goal = math.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            
            # Obstacle costs
            obs_costs = []
            for obs in self.static_obstacles:
                ox, oy = obs['position']
                r = obs['radius']
                dist = math.sqrt((x - ox)**2 + (y - oy)**2) - self.cfg.vehicle_radius - r
                # Approx cost for visualization
                margin_violation = 0.5 - dist 
                if margin_violation > 0:
                     cost_val = 2000.0 * (margin_violation ** 2)
                else:
                     cost_val = 0.0
                obs_costs.append({'pos': (ox, oy), 'dist': dist, 'cost': cost_val})
            
            debug_info['costs'] = {
                'goal_dist': dist_to_goal,
                'heading_error_deg': math.degrees(heading_error),
                'obstacle_costs': obs_costs,
                'total_solver_cost': self.solver.get_cost()
            }

        if solve_MPC_status != 0:
            # Try to get the solution anyway (might be suboptimal but usable)
            try:
                suboptimal_control = self.solver.get(0, 'u')
                return suboptimal_control, solve_MPC_status, debug_info
            except:
                # Fallback with intelligent obstacle avoidance steering
                fallback_control = self._compute_fallback_control(cur_state)
                return fallback_control, solve_MPC_status, debug_info

        optimal_control = self.solver.get(0, 'u')
        return optimal_control, solve_MPC_status, debug_info

    def _compute_fallback_control(self, cur_state):
        """
        Compute emergency fallback control when solver fails.
        Instead of just stopping, steer away from the nearest obstacle.
        """
        x, y, v, psi = cur_state
        
        # Find the nearest obstacle and compute avoidance direction
        min_dist = float('inf')
        nearest_obs_pos = None
        
        # Check static obstacles
        for obs in self.static_obstacles:
            ox, oy = obs['position']
            dist = math.hypot(x - ox, y - oy) - obs['radius'] - self.cfg.vehicle_radius
            if dist < min_dist:
                min_dist = dist
                nearest_obs_pos = (ox, oy)
        
        # Check dynamic obstacles
        for obs in self.dynamic_obstacles:
            ox, oy = obs['trajectory'][0], obs['trajectory'][1]  # current position
            dist = math.hypot(x - ox, y - oy) - obs['radius'] - self.cfg.vehicle_radius
            if dist < min_dist:
                min_dist = dist
                nearest_obs_pos = (ox, oy)
        
        # Compute steering direction away from obstacle
        if nearest_obs_pos is not None and min_dist < 5.0:  # Only if obstacle is close
            ox, oy = nearest_obs_pos
            
            # Vector from obstacle to car
            dx = x - ox
            dy = y - oy
            
            # Angle from obstacle to car
            angle_from_obs = math.atan2(dy, dx)
            
            # Compute the angle perpendicular to the obstacle (turn left or right)
            # Choose the direction that moves towards the goal
            goal_angle = math.atan2(self.goal[1] - y, self.goal[0] - x)
            
            # Two perpendicular directions
            perp_angle1 = angle_from_obs + math.pi / 2
            perp_angle2 = angle_from_obs - math.pi / 2
            
            # Choose the one closer to goal direction
            diff1 = abs(math.atan2(math.sin(goal_angle - perp_angle1), math.cos(goal_angle - perp_angle1)))
            diff2 = abs(math.atan2(math.sin(goal_angle - perp_angle2), math.cos(goal_angle - perp_angle2)))
            
            target_angle = perp_angle1 if diff1 < diff2 else perp_angle2
            
            # Compute steering to achieve target angle
            heading_error = target_angle - psi
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))  # normalize
            
            # Proportional steering control
            steer = np.clip(heading_error * 1.5, -self.cfg.max_steering_angle, self.cfg.max_steering_angle)
            
            # If very close to obstacle, also slow down; otherwise maintain some speed
            if min_dist < 1.0:
                accel = -1.0  # brake
            else:
                accel = 0.5  # gentle acceleration to keep moving
            
            return np.array([steer, accel])
        else:
            # No close obstacle, just steer towards goal
            goal_angle = math.atan2(self.goal[1] - y, self.goal[0] - x)
            heading_error = goal_angle - psi
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            steer = np.clip(heading_error * 1.0, -self.cfg.max_steering_angle, self.cfg.max_steering_angle)
            return np.array([steer, 0.5])
    









