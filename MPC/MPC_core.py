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
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # Real-time iteration for faster solving

        self.ocp.solver_options.qp_solver_iter_max = 200
        self.ocp.solver_options.nlp_solver_max_iter = 50  # fewer iterations for real-time
        self.ocp.solver_options.tol = 1e-3  # relaxed tolerance for stability
        self.ocp.solver_options.qp_solver_cond_N = 5  # partial condensing
        
        # Regularization for numerical stability
        self.ocp.solver_options.levenberg_marquardt = 1e-2
        
        self.solver = AcadosOcpSolver(self.ocp, json_file = 'vehicle_mpc.json')


    def setup_hard_constraints(self):   
        """
        Define constraints for the MPC problem.
        Static obstacles are now SOFT constraints with slack to prevent infeasibility.
        """
        
        self.ocp.constraints.x0 = ca.DM.zeros(self.nx)  # initial state constraint placeholder
        self.ocp.constraints.lbx = [self.cfg.min_speed]
        self.ocp.constraints.ubx = [self.cfg.max_speed]
        self.ocp.constraints.idxbx = [2]  # speed index

        self.ocp.constraints.lbu = [-self.cfg.max_steering_angle, self.cfg.max_deceleration]
        self.ocp.constraints.ubu = [self.cfg.max_steering_angle, self.cfg.max_acceleration]
        self.ocp.constraints.idxbu = [0, 1]  # steering angle and acceleration indices
        
        h_list = []

        for static_obs in self.static_obstacles:
            position = ca.DM(static_obs['position'])
            radius = static_obs['radius']
            # Add constraints to avoid static obstacles
            h_static = distance(self.states[0:2], position, radius1 = self.cfg.vehicle_radius, radius2 = radius)
            h_list.append(h_static)
        
        if len(h_list) > 0:
            h = ca.vertcat(*h_list)  # concatenate all h_i into a single vector

            self.ocp.model.con_h_expr = h
            # SOFT CONSTRAINTS: allow small violations with heavy penalty
            self.ocp.constraints.lh = np.zeros(len(h_list))      # h_i >= 0, no collision
            self.ocp.constraints.uh = np.ones(len(h_list)) * 1e6  # HUGE upper bound
            
            # Add slack variables for soft constraints
            # L2 penalty weight and L1 penalty weight for constraint violations
            self.ocp.cost.zl = 100.0 * np.ones(len(h_list))  # L1 penalty for lower bound violation
            self.ocp.cost.zu = 0.0 * np.ones(len(h_list))    # L1 penalty for upper bound violation  
            self.ocp.cost.Zl = 500.0 * np.ones(len(h_list))  # L2 penalty for lower bound violation
            self.ocp.cost.Zu = 0.0 * np.ones(len(h_list))    # L2 penalty for upper bound violation
            
            # Slack bounds (how much violation is allowed)
            self.ocp.constraints.lsh = np.zeros(len(h_list))  # slack lower bound
            self.ocp.constraints.ush = np.zeros(len(h_list))  # slack upper bound (0 means no constraint)
            self.ocp.constraints.idxsh = np.arange(len(h_list))  # indices of soft constraints


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
        """
        cost = 0

        # Weights for different cost components
        Q_obs = 30.0  # reduced - soft constraints handle hard avoidance
        Q_goal = 2.0  # increased for stronger goal attraction
        Q_terminal_goal = 30.0 # weight for terminal goal reaching
        Q_input_a = 0.01  # very low - allow aggressive acceleration
        Q_input_delta = 0.1  # increased to penalize large steering angles
        Q_velocity = 2.0  # weight for velocity
        Q_heading = 5.0  # strong heading cost to point towards goal

        safety_margin = 3.0  # meters - start avoiding when closer than this

        # Cost for static obstacle avoidance: smoother exponential barrier
        for static_obs in self.static_obstacles:
            position = ca.DM(static_obs['position'])
            radius = static_obs['radius']
            dist = distance(self.states[0:2], position, radius1=self.cfg.vehicle_radius, radius2=radius)
            # Smoother barrier: exponential decay instead of 1/dist
            # High cost when close, decays smoothly when far
            cost += Q_obs * ca.exp(-dist / 2.0)

        # Cost for reaching the goal (quadratic for stronger gradient)
        dis_to_goal_sq = (self.states[0] - self.goal[0])**2 + (self.states[1] - self.goal[1])**2
        cost += Q_goal * dis_to_goal_sq

        # Heading cost - STRONG penalty for not pointing towards goal
        goal_angle = ca.atan2(self.goal[1] - self.states[1], self.goal[0] - self.states[0])
        heading_error = goal_angle - self.states[3]  # psi
        # Normalize angle to [-pi, pi]
        heading_error = ca.atan2(ca.sin(heading_error), ca.cos(heading_error))
        cost += Q_heading * heading_error**2

        # Cost for input effort - penalize large steering to encourage straight driving when possible
        cost += Q_input_delta * self.controls[0]**2
        cost += Q_input_a * self.controls[1]**2

        # Velocity cost to encourage higher speeds
        v_ref = self.cfg.max_speed * 0.7  # target 70% of max speed
        cost += Q_velocity * (self.states[2] - v_ref)**2

        # Cost for dynamic obstacle avoidance - also smoother
        for i in range(self.num_dyn_obs):
            obs_x = self.ocp.model.p[i * 2]
            obs_y = self.ocp.model.p[i * 2 + 1]
            dist_dyn = distance(self.states[0:2], ca.vertcat(obs_x, obs_y),
                                radius1=self.cfg.vehicle_radius, radius2=self.dynamic_obstacles[i]['radius'])
            # Smoother exponential barrier
            cost += Q_obs * ca.exp(-dist_dyn / 2.0)

        # Transient cost type setting
        self.ocp.model.cost_expr_ext_cost = cost
        self.ocp.model.cost_type = 'EXTERNAL'

        # Terminal cost for final state - include heading towards goal
        terminal_cost = 0
        dis_to_goal_terminal_sq = (self.states[0] - self.goal[0])**2 + (self.states[1] - self.goal[1])**2
        terminal_cost += Q_terminal_goal * dis_to_goal_terminal_sq
        
        # Terminal heading cost - ensure vehicle points towards goal at end of horizon
        terminal_heading_error = ca.atan2(ca.sin(goal_angle - self.states[3]), ca.cos(goal_angle - self.states[3]))
        terminal_cost += 10.0 * terminal_heading_error**2

        self.ocp.model.cost_expr_ext_cost_e = terminal_cost
        self.ocp.model.cost_type_e = 'EXTERNAL'


    def solve(self, cur_state):
        """
        Solve the MPC problem for the current state and dynamic obstacle positions.
        Returns: (control, solver_status)
        """
        self.solver.set(0, 'lbx', cur_state) # cur_state: [x, y, v, psi] numpy array
        self.solver.set(0, 'ubx', cur_state)

        # Set dynamic obstacle positions as parameters in all horizon steps
        for i in range(self.N):
            obs_pos_i = self.get_predicted_obs_trajectories(i)
            p_i = obs_pos_i
            self.solver.set(i, 'p', p_i)

        solve_MPC_status = self.solver.solve()

        if solve_MPC_status != 0:
            # Try to get the solution anyway (might be suboptimal but usable)
            try:
                suboptimal_control = self.solver.get(0, 'u')
                return suboptimal_control, solve_MPC_status
            except:
                # Fallback with intelligent obstacle avoidance steering
                fallback_control = self._compute_fallback_control(cur_state)
                return fallback_control, solve_MPC_status

        optimal_control = self.solver.get(0, 'u')
        return optimal_control, solve_MPC_status

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
    









