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
    def __init__(self, static_obstacles, dynamic_obstacles, horizon = 20, dt = 0.1):
        self.solver = None
        self.ocp = None

        self.nx = 4       # number of states
        self.nu = 2       # number of inputs

        # self.car_axis_param = car_axis_param # l_f and l_r of the vehicle
        # self.goal = goal    # goal position

        self.N = horizon    # prediction horizon
        self.dt = dt        # time step
        self.cfg = cfg      # configuration file
        self.goal = ca.DM(self.cfg.goal)
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

        num_params = self.num_dyn_obs * 2 # dynamic obstacles: position [x, y] at a given time step
        p = ca.SX.sym('p', num_params)
        
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
        self.ocp.solver_options.nlp_solver_type = 'SQP'  # Full SQP for better convergence

        self.ocp.solver_options.qp_solver_iter_max = 600
        self.ocp.solver_options.nlp_solver_max_iter = 500  # More iterations for convergence
        self.ocp.solver_options.tol = 1e-5  # Tighter tolerance
        self.ocp.solver_options.qp_solver_cond_N = 5  # partial condensing
        
        # Regularization for numerical stability
        self.ocp.solver_options.levenberg_marquardt = 1e-2
        
        self.solver = AcadosOcpSolver(self.ocp, json_file = 'vehicle_mpc.json')


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
        Balanced configuration: strong goal seeking + effective obstacle avoidance
        """
        cost = 0

        # Weights - BALANCED configuration with stronger obstacle avoidance
        Q_obs = 500.0  # HIGH obstacle avoidance - critical for safety
        Q_close_penalty = 2000.0  # Penalty for being too close to obstacles
        Q_goal = 10.0  # Goal attraction - steady pull  
        Q_terminal_goal = 20.0 # Terminal goal - very important
        Q_input_a = 0.1  # Allow acceleration changes
        Q_input_delta = 0.5  # Allow steering for avoidance
        Q_velocity = 500  # Maintain speed
        Q_heading = 2.0  # Heading towards goal
        Q_terminal_heading = 20.0  # Terminal heading towards goal

        # Goal direction (computed once for reuse)
        goal_angle = ca.atan2(self.goal[1] - self.states[1], self.goal[0] - self.states[0])
        
        # Cost for reaching the goal
        dis_to_goal_sq = (self.states[0] - self.goal[0])**2 + (self.states[1] - self.goal[1])**2
        dis_to_goal = ca.sqrt(dis_to_goal_sq + 0.01)
        cost += Q_goal * dis_to_goal

        # Heading cost - penalize deviation from goal direction
        heading_error = goal_angle - self.states[3]  # psi
        heading_error_normalized = ca.atan2(ca.sin(heading_error), ca.cos(heading_error))
        cost += Q_heading * heading_error_normalized**2

        # Cost for static obstacle avoidance: use barrier-like function
        # Scale parameter controls the range of influence
        obs_scale = 3.0  # Larger scale = longer range influence
        safety_margin = 1.5  # minimum safe distance from obstacle surface
        for static_obs in self.static_obstacles:
            position = ca.DM(static_obs['position'])
            radius = static_obs['radius']
            dist = distance(self.states[0:2], position, radius1=self.cfg.vehicle_radius, radius2=radius)
            # Exponential barrier: becomes very large when dist approaches 0
            cost += Q_obs * ca.exp(-dist / obs_scale)   # TODO: cost too low?
            # cost += Q_obs * (1/(dist + 1e-3)**4)  # Inverse distance cost
            
            # Soft constraint: quadratic penalty when closer than safety margin
            violation = ca.fmax(0, safety_margin - dist)    # TODO: to be smoothened?
            cost += Q_close_penalty * violation**2

        # Cost for dynamic obstacle avoidance
        for i in range(self.num_dyn_obs):
            obs_x = self.ocp.model.p[i * 2]
            obs_y = self.ocp.model.p[i * 2 + 1]
            dist_dyn = distance(self.states[0:2], ca.vertcat(obs_x, obs_y),
                                radius1=self.cfg.vehicle_radius, radius2=self.dynamic_obstacles[i]['radius'])
            cost += Q_obs * ca.exp(-dist_dyn / obs_scale)

            violation_dyn = ca.fmax(0, safety_margin - dist_dyn)    # TODO: to be smoothened?
            cost += Q_close_penalty * violation_dyn**2

        # Cost for input effort
        cost += Q_input_delta * self.controls[0]**2  # steering
        cost += Q_input_a * self.controls[1]**2  # acceleration

        # Velocity cost - encourage steady forward motion
        v_ref = 0.4 * cfg.max_speed  # target speed 
        cost += Q_velocity * (self.states[2] - v_ref)**2

        # Transient cost type setting
        self.ocp.model.cost_expr_ext_cost = cost
        self.ocp.model.cost_type = 'EXTERNAL'

        # Terminal cost for final state - VERY STRONG goal attraction
        terminal_cost = 0
        dis_to_goal_terminal = ca.sqrt((self.states[0] - self.goal[0])**2 + (self.states[1] - self.goal[1])**2 + 0.01)
        terminal_cost += Q_terminal_goal * dis_to_goal_terminal
        
        # Terminal heading cost - should point towards goal
        terminal_heading_error = ca.atan2(ca.sin(goal_angle - self.states[3]), ca.cos(goal_angle - self.states[3]))
        terminal_cost += Q_terminal_heading * terminal_heading_error**2

        self.ocp.model.cost_expr_ext_cost_e = terminal_cost
        self.ocp.model.cost_type_e = 'EXTERNAL'


    def solve(self, cur_state, debug=False):
        """
        Solve the MPC problem for the current state and dynamic obstacle positions.
        Returns: (control, solver_status, debug_info)
        """
        # Set initial state constraint
        self.solver.set(0, 'lbx', cur_state) # cur_state: [x, y, v, psi] numpy array
        self.solver.set(0, 'ubx', cur_state)
        
        # Initialize trajectory with obstacle-aware path
        x0, y0, v0, psi0 = cur_state
        goal_x, goal_y = float(self.goal[0]), float(self.goal[1])
        
        # Compute direction to goal
        goal_angle = math.atan2(goal_y - y0, goal_x - x0)
        v_init = max(v0, 2.0)
        
        # Check possible collision on straight path to goal
        def check_collision(x, y, step_i):
            """Check if point (x,y) is too close to any obstacle"""
            min_safe_dist = 2.0  # safety margin
            for obs in self.static_obstacles:
                ox, oy = obs['position']
                r = obs['radius']
                dist = math.sqrt((x - ox)**2 + (y - oy)**2) - r - self.cfg.vehicle_radius
                if dist < min_safe_dist:
                    return True, (ox, oy, r)
                
            current_time = step_i * self.dt

            for obs in self.dynamic_obstacles:
                curr_x, curr_y, vx, vy = obs['trajectory']
                pred_ox = curr_x + vx * current_time
                pred_oy = curr_y + vy * current_time
                r = obs['radius']
                
                dist = math.sqrt((x - pred_ox)**2 + (y - pred_oy)**2) - r - self.cfg.vehicle_radius
                if dist < min_safe_dist:
                    return True, (pred_ox, pred_oy, r)
                
            return False, None
        
        # Find first collision point on straight path
        collision_found = False
        collision_obs = None
        for i in range(self.N + 1):
            t = i * self.dt
            x_check = x0 + v_init * math.cos(goal_angle) * t
            y_check = y0 + v_init * math.sin(goal_angle) * t
            has_collision, obs_info = check_collision(x_check, y_check, i)

            if has_collision:
                collision_found = True
                collision_obs = obs_info
                break
        
        if collision_found and collision_obs:
            # Generate path that goes around the obstacle
            ox, oy, obs_r = collision_obs
            
            # Determine which side to go around (left or right of obstacle)
            # Use cross product to decide: if goal is to the left, go left
            dx_to_obs = ox - x0
            dy_to_obs = oy - y0
            dx_to_goal = goal_x - x0
            dy_to_goal = goal_y - y0
            cross = dx_to_obs * dy_to_goal - dy_to_obs * dx_to_goal
            
            # Tangent direction around obstacle
            avoidance_angle = goal_angle + (math.pi/4 if cross > 0 else -math.pi/4)
            
            # Initialize with curved path
            for i in range(self.N + 1):
                t = i * self.dt
                # Blend between avoidance and goal direction
                blend = min(1.0, t / 3.0)  # transition over 3 seconds
                current_angle = avoidance_angle * (1 - blend) + goal_angle * blend
                x_init = x0 + v_init * math.cos(current_angle) * t
                y_init = y0 + v_init * math.sin(current_angle) * t
                self.solver.set(i, 'x', np.array([x_init, y_init, v_init, current_angle]))
            
            # Initialize controls for avoidance
            for i in range(self.N):
                steer_init = (avoidance_angle - psi0) * 0.1 if i < 10 else 0.0
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

        # Set dynamic obstacle positions as parameters in all horizon steps
        for i in range(self.N):
            obs_pos_i = self.get_predicted_obs_trajectories(i)
            p_i = obs_pos_i
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
            goal_x, goal_y = float(self.goal[0]), float(self.goal[1])
            
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
                cost_val = 50.0 * math.exp(-dist / 1.0)
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
    









