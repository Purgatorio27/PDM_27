import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from Obstacles import distance
import Config as cfg
import numpy as np

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
        # self.ocp.parameter_values = np.zeros((num_params,))

        self.setup_hard_constraints()
        self.add_costs()

        # Solver options
        self.ocp.solver_options.tf = self.N * self.dt
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'DISCRETE'
        self.ocp.solver_options.nlp_solver_type = 'SQP'

        self.ocp.solver_options.qp_solver_iter_max = 500         # TODO: tuning
        self.ocp.solver_options.nlp_solver_max_iter = 150
        self.ocp.solver_options.tol = 1e-5
        
        self.solver = AcadosOcpSolver(self.ocp, json_file = 'vehicle_mpc.json')


    def setup_hard_constraints(self):   
        """
        Define hard constraints for the MPC problem.
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
        
        h = ca.vertcat(*h_list)  # concatenate all h_i into a single vector

        self.ocp.model.con_h_expr = h
        self.ocp.constraints.lh = np.zeros(len(h_list))      # h_i >= 0, no collision
        self.ocp.constraints.uh = np.ones(len(h_list)) * 1e6  # HUGE upper bound


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
            

    def add_costs(self):                                     # TODO : tuning for better performance
        """
        Define the cost function for the MPC problem.
        """
        cost = 0

        # Weights for different cost components
        Q_obs = 10.0  # weight for obstacle avoidance
        Q_goal = 10.0  # weight for goal reaching

        Q_terminal_goal = 5.0 # weight for terminal goal reaching
    
        Q_input_a = 0.5  # weight for acceleration input
        Q_input_delta = 0.5 # weight for steering input
        Q_velocity = 4.0  # weight for velocity 

        # Cost for static obstacle avoidance
        for static_obs in self.static_obstacles:
            position = ca.DM(static_obs['position'])
            radius = static_obs['radius']
            dist = distance(self.states[0:2], position, radius1 = self.cfg.vehicle_radius, radius2 = radius)
            cost += Q_obs * 1/(dist**2 + 1e-6)  # avoid division by zero
         
        # Cost for reaching the goal
        dis_to_goal = ca.sqrt((self.states[0] - self.goal[0])**2 + (self.states[1] - self.goal[1])**2)
        cost += Q_goal * dis_to_goal**2

        # Cost for input effort
        cost += Q_input_delta * self.controls[0]**2
        cost += Q_input_a * self.controls[1]**2
        
        # Velocity cost to encourage higher speeds
        v_ref = self.cfg.max_speed 
        cost += Q_velocity * (self.states[2] - v_ref)**2
     
        # Cost for dynamic obstacle avoidance
        for i in range(self.num_dyn_obs):
            
            obs_x = self.ocp.model.p[i * 2]
            obs_y = self.ocp.model.p[i * 2 + 1]
            
            dist_dyn = distance(self.states[0:2], ca.vertcat(obs_x, obs_y), \
                                radius1 = self.cfg.vehicle_radius, radius2 = self.dynamic_obstacles[i]['radius'])
            cost += Q_obs * 1/(dist_dyn**2 + 1e-6)
        
        # Transient cost type setting
        self.ocp.model.cost_expr_ext_cost = cost
        self.ocp.model.cost_type = 'EXTERNAL'

        # Terminal cost for final state
        terminal_cost = 0
        dis_to_goal_terminal = ca.sqrt((self.states[0] - self.goal[0])**2 + (self.states[1] - self.goal[1])**2)
        terminal_cost += Q_terminal_goal * dis_to_goal_terminal**2

        self.ocp.model.cost_expr_ext_cost_e = terminal_cost
        self.ocp.model.cost_type_e = 'EXTERNAL'


    def solve(self, cur_state):   
        """
        Solve the MPC problem for the current state and dynamic obstacle positions.
        (Simulated perception can only provide current positions and velocities of dynamic obstacles)
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
            print('MPC solver failed with this status:', solve_MPC_status)
            return np.array([0.0, self.cfg.max_deceleration])  # safe fallback action
            
        optimal_control = self.solver.get(0, 'u')
        print('Optimal control:', optimal_control)

        return optimal_control
    









