import numpy as np
import casadi as ca
import casadi.tools as ctools
import time
import random
import pprint
import copy

from controllers.tools.control_allocator import ControlAllocator
from controllers.tools.input_bounds import InputBounds
from controllers.tools.spiral_parameters import SpiralParameters
from controllers.tools.terminal_ingredients import load_terminal_ingredients
from util.utils import RotCasadi, RotFull, RotFullInv
from util.get_trajectory import load_trajectory
from util.controller_debug import ControllerDebug, DebugVal, Logger

# RobotToCenterRot = Rot3Inv 
# CenterToRobotRot = Rot3
RobotToCenterRot = RotFullInv
CenterToRobotRot = RotFull

class SpiralingController:
    """
    Controller implementing micro-orbiting
    """
    def __init__(self, model, params, debug):
        self.model = model
        self.params = params
        self.debug = debug
        self.logger = Logger()

        self.spiral_params = SpiralParameters(model)
        self.set_model(model)

        self.trajectory = None

        self.Nt = self.params["horizon"]

        # Initialize variables
        self.set_cost_functions()

        self.build_solver()
        self.optimal_solution = None

    def set_model(self, model):
        self.model = copy.deepcopy(model)
        self.dynamics = self.model.dynamics
        self.bounds = InputBounds(self.model)
        self.contr_alloc = ControlAllocator(self.model, self.bounds)

        self.mass = model.mass
        self.J = model.inertia
        self.dt = model.dt
        self.Nx, self.Nu = model.Nx, model.Nu
        # Number of optimized states
        self.Nopt = 9 # pos, vel, omega uncontrolled: q

    def set_cost_functions(self):
        """
        Helper method to create CasADi functions for the MPC cost objective.
        """
        # Create functions and function variables for calculating the cost
        Q = ca.MX.sym('Q', self.Nopt, self.Nopt)
        P = ca.MX.sym('P', self.Nopt, self.Nopt)
        R = ca.MX.sym('R', self.Nu, self.Nu)

        x = ca.MX.sym('x', self.Nopt)
        xr = ca.MX.sym('xr', self.Nopt)
        u = ca.MX.sym('u', self.Nu)

        # Prepare variables
        e_vec = x - xr

        # Calculate running cost
        ln = ca.mtimes(ca.mtimes(e_vec.T, Q), e_vec) + ca.mtimes(ca.mtimes(u.T, R), u)

        self.running_cost = ca.Function('ln', [x, xr, Q, u, R], [ln])

        # Calculate terminal cost
        self.terminal_cost, self.terminal_set = load_terminal_ingredients("./terminal.yaml")

        e = ca.MX.sym("e", 9)
        # print(self.terminal_cost(*ca.vertsplit(e)))
        # breakpoint()

    def build_solver(self):
        build_solver_start = time.time()

        # Cost function weights
        param_set = self.params[self.params["param_set"]]
        Q = np.diag(param_set["Q"])
        R = np.diag(param_set["R"])

        self.Q = ca.MX(Q)
        self.R = ca.MX(R)

        self.x_sp = None
        self.u_sp = None

        x0 = ca.MX.sym('x0', self.Nx)

        # create vector representing the reference trajectory
        x_ref = ca.reshape(ca.MX.sym('x_ref', self.Nopt, (self.Nt+1)), (-1, 1))
        u_ref = ca.reshape(ca.MX.sym('u_ref', self.Nu, (self.Nt+1)), (-1, 1))

        param_s = ca.vertcat(x0, x_ref, u_ref)

        # Create optimization variables
        opt_var = ctools.struct_symMX([(ctools.entry('u', shape=(self.Nu,), repeat=self.Nt),
                                        ctools.entry('x', shape=(self.Nx,), repeat=self.Nt+1),
                                        )])
        self.opt_var = opt_var
        self.num_var = opt_var.size

        # Set initial values
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
        con_eq.append(opt_var['x', 0] - x0)    

        # Decision variable boundries
        self.optvar_lb = opt_var(-np.inf)
        self.optvar_ub = opt_var(np.inf)

        # Bounds on x, default is None
        xub = self.params.get("xub", None)
        xlb = self.params.get("xlb", None)

        # Bounds on u in rob_local system
        ChullMat, ChullVec = self.bounds.get_conv_hull() # A, b
        # But calculate instead in force-aligned system
        beta = self.spiral_params.beta
        # ChullMat = ChullMat @ CenterToRobotRot(beta - np.pi/2)
        ChullMat = ChullMat @ CenterToRobotRot(beta)

        # Compensating input to get the virtual incontrollable force, not the pysical one
        # u_comp = RobotToCenterRot(beta - np.pi/2) @ self.spiral_params.compensation_force
        u_comp = RobotToCenterRot(beta) @ self.spiral_params.compensation_force
        self.u_comp = u_comp
        # Physical uncontrollable force
        # u_uncontrolled = RobotToCenterRot(beta - np.pi/2) @ self.model.faulty_input_simple.flatten()
        u_uncontrolled = RobotToCenterRot(beta) @ self.model.faulty_force_generalized.flatten()
        u_uncontrolled = u_uncontrolled.flatten()

        # Generate MPC Problem
        for t in range(self.Nt):
            # Get variables
            x_t = opt_var['x', t]

            x_r = x_ref[t*self.Nopt:(t+1)*self.Nopt]
            u_r = u_ref[t*self.Nu:(t+1)*self.Nu]
            # The saved nominal input is not yet corrected for the orientation, correct now...
            alpha = x_t[9:13]
            rotInvCa = RotCasadi(alpha).T
            # rot = ca.MX(2,2)
            # rot[0, 0] = ca.cos(alpha)
            # rot[0, 1] = ca.sin(alpha)
            # rot[1, 0] = -ca.sin(alpha)
            # rot[1, 1] = ca.cos(alpha)
            u_r = ca.vcat([
                ca.mtimes(rotInvCa, u_r[0:3]), 
                u_r[3:6]
            ])

            u_t = opt_var['u', t]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t + u_r + u_comp) 
            con_eq.append(x_t_next - opt_var['x', t + 1])

            # Input constraints
            con_ineq.append(ChullMat @ (u_t + u_r + u_comp + u_uncontrolled))
            con_ineq_ub.append(ChullVec) # u_comp taken into account here
            con_ineq_lb.append(np.full(ChullVec.shape, -ca.inf))

            # State constraints
            if not(xub is None and xlb is None):
                xub = np.full((self.Nx,), ca.inf) if xub is None else xub
                xlb = np.full((self.Nx,), -ca.inf) if xlb is None else xlb
                con_ineq.append(x_t)
                con_ineq_ub.append(xub)
                con_ineq_lb.append(xlb)

            # Objective Function / Cost Function
            obj += self.running_cost(x_t[0:self.Nopt], x_r, self.Q, u_t, self.R)

        # Terminal ingredients
        x_t = opt_var['x', self.Nt]
        x_r = x_ref[self.Nt*self.Nopt:(self.Nt+1)*self.Nopt]

        # Terminal Cost
        e_N = x_t[0:self.Nopt] - x_r
        obj += self.terminal_cost(*ca.vertsplit(e_N))

        # Terminal Constraint
        con_t = self.terminal_set
        con_ineq.append(ca.mtimes(con_t.A, e_N))
        con_ineq_lb.append(-ca.inf * np.ones_like(con_t.b))
        con_ineq_ub.append(con_t.b)

        # Equality constraints are reformulated as inequality constraints with 0<=g(x)<=0
        # -> Refer to CasADi documentation: NLP solver only accepts inequality constraints
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con,))
        con_eq_ub = np.zeros((num_eq_con,))

        # Set constraints
        con = ca.vertcat(*(con_eq + con_ineq))
        self.con_lb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.con_ub = ca.vertcat(con_eq_ub, *con_ineq_ub)

        # Build NLP Solver (can also solve QP)
        nlp = dict(x=opt_var, f=obj, g=con, p=param_s)
        options = { # maybe use jit compiler?
            # 'ipopt.print_level': 5,
            'ipopt.print_level': 0,
            'ipopt.tol': 1e-3,
            'ipopt.check_derivatives_for_naninf': "yes",
            'print_time': False,
            'verbose': True,
            'expand': True
        }
        solver_opts = self.params.get("solver_opts", None)
        if solver_opts is not None:
            options.update(solver_opts)
        self.solver = ca.nlpsol('spiral_MPC_sol', 'ipopt', nlp, options)

        self.logger.info('\n________________________________________')
        self.logger.info(f"# Time to build mpc solver: {time.time() - build_solver_start} sec")
        self.logger.info(f"# Number of variables: {self.num_var}")
        self.logger.info(f"# Number of equality constraints: {num_eq_con}")
        self.logger.info(f"# Number of inequality constraints: {num_ineq_con}")
        self.logger.info(f"# Horizon steps: {self.Nt * self.dt}s into the future")
        self.logger.info('----------------------------------------')

    def load_trajectory(self, cmd, duration, fpath=None):
        """
        Load/generate a trajectory for the controller to track.

        Args:
            cmd (str): Action to perform, see file util/get_trajectory.py for details
            duration (int): Duration of the trajectory
            fpath (str): Path to the trajectory file, defaults to None
        
        Returns:
            ndarray: Trajectory
        """
        traj = load_trajectory(cmd, self.dt, duration, file_path=fpath)
        self.assign_trajectory(traj)

    def assign_trajectory(self, traj):
        """
        Assign a trajectory to the controller. Also calculates the necessary inputs to realize this 
        trajectory.

        Args:
            traj (ndarray): Trajectory to assign
        """
        # Prolong the trajectory to prevent the controller from running out of points 
        original_traj = np.hstack((traj, np.tile(traj[:, -1:], (1, self.Nt))))

        # Original traj in the form [pos, vel, q, omega]
        # New traj in form [pos_c, vel_c, omega_c]

        omega_des = np.tile(
            self.spiral_params.omega_des, 
            (np.size(original_traj, axis=1),1)
        ).T

        self.trajectory = np.concatenate((
            original_traj[0:6, :],
            omega_des
        ))

        # Calculate the nominal input that realizes this trajectory
        # In this controller, this is not compensated for the angle as the angle is arbitraty
        # This has to be taken into account later on!
        x = self.trajectory[0:3, :]
        secondDer = np.gradient(np.gradient(x, axis=1), axis=1) / self.dt**2
        # include the mass (not inerita because omega_dot=0)
        necessary_force = np.vstack((secondDer * self.mass, np.zeros_like(secondDer)))
        self.nominal_input = necessary_force

    def get_control(self, x0, t):
        # Convert robot state to orbit center state
        c0 = self.model.robot_to_center(x0)
        c0 = np.array(c0).flatten()

        # Prepare the reference trajectory
        x_ref, u_ref = self.get_next_trajectory_part(t)
        self.x_sp = x_ref.reshape(-1, 1, order='F')
        self.u_sp = u_ref.reshape(-1, 1, order='F')
        
        # Solve the optimization problem
        c, u, slv_time, cost, slv_status = self.solve_mpc(c0)

        u_nom_alpha_corrected = (RotFullInv(c0[9:13]) @ self.u_sp[0:self.Nu]).flatten()
        u_res = np.array(u[0]).flatten() + u_nom_alpha_corrected + self.u_comp
        
        beta = self.spiral_params.beta
        # u_res = CenterToRobotRot(beta - np.pi/2) @ u_res
        u_res = CenterToRobotRot(beta) @ u_res
        u_phys = self.contr_alloc.get_physical_input(u_res)

        debug = DebugVal(self, t)
        debug.set_state(x0)
        debug.set_circle_state(c0)
        debug.set_input(u_phys, self.model)
        debug.set_desired_state(self.x_sp[0:self.Nopt, 0])
        debug.calculate_errors()
        self.debug.add_debug_val(debug)

        return u_phys

    def solve_mpc(self, x0):
        solver_start = time.time()
        self.optvar_x0 = np.full((1, self.Nx), x0.T)

        # Initialize variables
        if self.optimal_solution is not None:
            # Initial guess of the warm start variables
            self.optvar_init['x'] = self.optimal_solution['x'][1:] + [ca.DM([0] * self.Nx)]
            # self.optvar_init['x'] = self.optimal_solution['x'][1:] + [ca.DM([0]*6 + [1e-5] * 3 + [0]*4)]
            self.optvar_init['u'] = self.optimal_solution['u'][1:] + [ca.DM([0] * self.Nu)]
        else:
            # Initialize with zero if no previous solution is available
            self.optvar_init = self.opt_var(0)
            # for i in range(self.Nt+1):
            #     self.optvar_init['x', i] = ca.DM([0]*6 + [0.1]*3 + [0]*4)
        self.optvar_init['x', 0] = self.optvar_x0[0]

        param = ca.vertcat(x0, self.x_sp, self.u_sp)
        
        args = dict(x0=self.optvar_init,
                    lbx=self.optvar_lb,
                    ubx=self.optvar_ub,
                    lbg=self.con_lb,
                    ubg=self.con_ub,
                    p=param)

        # Solve NLP
        sol = self.solver(**args)
        status = self.solver.stats()['return_status']
        optvar = self.opt_var(sol['x'])
        self.optimal_solution = optvar

        slv_time = time.time() - solver_start
        self.logger.info(f"MPC - CPU time: {slv_time:,.7f} seconds  |  Cost: {float(sol['f']):9.2f}  |  Horizon length: {self.Nt}  |  {status}")

        return optvar['x'], optvar['u'], slv_time, float(sol['f']), status

    def get_next_trajectory_part(self, t):
        """
        Get the next points in the trajectory that lies within the prediction horizon.
        """
        id_s = int(t / self.dt)
        id_e = id_s + self.Nt + 1
        x_r = self.trajectory[:, id_s:id_e]
        u_r = self.nominal_input[:, id_s:id_e]

        return x_r, u_r