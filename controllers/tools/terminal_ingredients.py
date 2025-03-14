import itertools
import numpy as np
import casadi as ca
import sympy as sp
import scipy.linalg as la
import yaml
import json

from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem
from pympc.control.controllers import ModelPredictiveController

from util.polytope import MyPolytope
from util.utils import RotCasadi, RotFull, RotFullInv
from controllers.tools.input_bounds import InputBounds

RobotToCenterRot = RotFullInv
CenterToRobotRot = RotFull

class explicitMPCTerminalIngredients:
    def __init__(self, model, tuning):
        self.model = model

        self.max_acceleration = tuning["max_acceleration"]
        self.k_omega = np.diag(tuning["k_omega"])
        self.tuning = tuning

        self.spiral_params = model.spiral_params

        self.r = self.spiral_params.r
        self.omega_des = self.spiral_params.omega_des
        self.virt_force = ca.DM(self.spiral_params.f_virt)

        self.mass = model.mass
        self.J = model.inertia
        self.dt = model.dt

        # The cost matrices for the function used in the MPC; Used to calculate terminal cost
        self.Q = np.diag(tuning["Q"])
        self.R = np.diag(tuning["R"])
        self.ensure_diagonal(self.Q)
        self.ensure_diagonal(self.R)

        self.M = self.spiral_params.M
        self.Minv = np.linalg.inv(self.M)

        self.Qu_tilde = self.Minv.T @ self.R @ self.Minv

    def ensure_diagonal(self, matrix):
        """
        Ensure that the matrix is diagonal.
        """
        if np.any( np.abs(matrix - np.diag(np.diag(matrix))) >= 0.0001* np.ones_like(matrix)):
            raise ValueError("Matrix is not diagonal.")

    def calc_empc_input_bounds(self):
        bounds = InputBounds(self.model)

        J = self.J; r = self.r; omega_des = self.omega_des; k_omega = ca.DM(self.k_omega)
        f_virt = RobotToCenterRot(self.spiral_params.beta) @ np.concatenate((self.virt_force, np.zeros_like(self.virt_force))) # Calculate in virt_force-aligned local system

        # First step: Take acceleration into account
        A, b = bounds.get_conv_hull()
        A = A @ CenterToRobotRot(self.spiral_params.beta) # Calculate in virt_force-aligned local system

        # transform to accelerations
        P_full = MyPolytope(A @ self.Minv, b) # This is definitely correct: Transform u->F, then check if it's in polytope
        P_offsetfree = P_full.minkowski_subtract_circle(self.max_acceleration)

        # Second step: Calculate upper bound on feedback-linearization
        opti = ca.Opti()
        bound = opti.variable(4)

        emax = bound[:3]
        r_empc = bound[3]

        constraints = []

        # r_empc, emaxi >= 0
        for i in range(max(bound.shape)):
            constraints.append(bound[i] >= 0)

        # for i in range(len(P_offsetfree.A)):
        #     a_i = ca.DM(P_offsetfree.A[i,:])
        #     b_i = ca.DM(P_offsetfree.b[i])
        #     constraints.append(
        #         ca.norm_2(a_i[:3] * r_empc) + a_i[3:].T @ k_omega @ emax <= b_i
        #     )
        #     constraints.append(
        #         ca.norm_2(a_i[:3] * r_empc) + a_i[3:].T @ (-1*k_omega) @ emax <= b_i
        #     )

        emax1 = emax[0]
        emax2 = emax[1]
        emax3 = emax[2]
        j00 = J[0,0].item()
        j11 = J[1,1].item()
        j22 = J[2,2].item()

        # create corner points of a box centered at zero with edge_length = 2u
        box_corners = list(itertools.product(*[[-emax[i], emax[i]] for i in range(3)])) 

        # The upper bound on the feedback-linearization terms
        rNorm = np.linalg.norm(r)
        om_d = np.linalg.norm(omega_des)

        upper_bound_fb_lin = [
            (-((emax3 - om_d)**2*j22**2 + emax2**2*j11**2)**2*j00**8 + 4*((emax3 - om_d)**2*j22**3 + emax2**2*j11**3)*((emax3 - om_d)**2*j22**2 + emax2**2*j11**2)*j00**7 + (-6*(emax3 - om_d)**4*j22**6 - 4*(emax3 - om_d)**2*j11**2*(emax3**2*rNorm**2 - 2*emax3*rNorm**2*om_d + 1/2*emax2**2)*j22**4 - 8*emax2**2*j11**3*(emax3 - om_d)**2*j22**3 - 4*((rNorm**2 + 1/2)*emax3**2 + (-2*rNorm**2 - 1)*om_d*emax3 + om_d**2/2)*emax2**2*j11**4*j22**2 - 6*emax2**4*j11**6)*j00**6 + 8*((emax3 - om_d)**2*j22**3 + emax2**2*j11**3)*((emax3 - om_d)**2*j22**4/2 + rNorm**2*emax3*j11**2*(emax3 - 2*om_d)*j22**2 + emax2**2*j11**4/2)*j00**5 + (-(emax3 - om_d)**4*j22**8 - 4*rNorm**2*emax3*j11**2*(emax3 - 2*om_d)*(emax3 - om_d)**2*j22**6 - 2*emax2**2*j11**4*(emax3 - om_d)**2*(rNorm**2 + 1)*j22**4 + 4*rNorm**2*emax2**2*j11**6*((rNorm**2 - 1)*emax3**2 + (-2*rNorm**2 + 2)*om_d*emax3 + rNorm**2*om_d**2 - emax2**2/2)*j22**2 - emax2**4*j11**8)*j00**4 + 4*rNorm**2*j22**2*((emax3 - om_d)**2*j22**3 + emax2**2*j11**3)*emax2**2*j11**4*j00**3 - 4*rNorm**2*j22**2*emax2**2*j11**4*(-(emax3 - om_d)**2*j22**4/2 + 2*j11*(emax3 - om_d)**2*j22**3 + ((rNorm**2 - 1)*emax3**2 + (-2*rNorm**2 + 2)*om_d*emax3 - om_d**2)*j11**2*j22**2 + emax2**2*j11**4/2)*j00**2 - emax2**4*j11**8*j22**4*rNorm**4)/(4*j11**4*j00**4*j22**4*rNorm**2),
            (-(emax1**2*j00**2 + emax2**2*j11**2)**2*j22**8 + 4*(emax1**2*j00**3 + emax2**2*j11**3)*(emax1**2*j00**2 + emax2**2*j11**2)*j22**7 + (-6*emax1**4*j00**6 - 4*(rNorm**2*emax1**2 - rNorm**2*om_d**2 + 1/2*emax2**2)*emax1**2*j11**2*j00**4 - 8*emax1**2*emax2**2*j00**3*j11**3 - 4*((rNorm**2 + 1/2)*emax1**2 - rNorm**2*om_d**2)*j11**4*emax2**2*j00**2 - 6*emax2**4*j11**6)*j22**6 + 8*(emax1**2*j00**4/2 + rNorm**2*j11**2*(emax1 - om_d)*(emax1 + om_d)*j00**2 + emax2**2*j11**4/2)*(emax1**2*j00**3 + emax2**2*j11**3)*j22**5 + (-emax1**4*j00**8 - 4*rNorm**2*emax1**2*j11**2*(emax1 - om_d)*(emax1 + om_d)*j00**6 - 2*emax1**2*emax2**2*j11**4*(rNorm**2 + 1)*j00**4 + 4*rNorm**2*(-emax2**2/2 + (rNorm**2 - 1)*emax1**2 + om_d**2)*j11**6*emax2**2*j00**2 - emax2**4*j11**8)*j22**4 + 4*rNorm**2*emax2**2*j00**2*j11**4*(emax1**2*j00**3 + emax2**2*j11**3)*j22**3 - 4*rNorm**2*j00**2*(-emax1**2*j00**4/2 + 2*emax1**2*j00**3*j11 + ((rNorm**2 - 1)*emax1**2 - rNorm**2*om_d**2)*j11**2*j00**2 + emax2**2*j11**4/2)*j11**4*emax2**2*j22**2 - emax2**4*j00**4*j11**8*rNorm**4)/(4*j11**4*j00**4*j22**4*rNorm**2),
            rNorm**2*(emax1**2 + emax3**2 - 2*emax3*om_d)**2 + emax1**2*(-emax3 + om_d)**2*(j00 - j22)**2/j11**2,
            rNorm**2*emax3**2*(-emax3 + 2*om_d)**2 + emax2**2*rNorm**2*j11**2*(-emax3 + om_d)**2/j22**2 + emax2**2*(-emax3 + om_d)**2*(j11 - j22)**2/j00**2,
            emax2**2*emax1**2*rNorm**2*j11**2/j00**2 + rNorm**2*(emax1**2 - om_d**2)**2 + emax2**2*emax1**2*(j00 - j11)**2/j22**2,
            emax2**2*emax1**2*rNorm**2*j11**2/j00**2 + rNorm**2*(emax1**2 + emax3**2 - 2*emax3*om_d)**2 + emax2**2*rNorm**2*j11**2*(-emax3 + om_d)**2/j22**2 + emax2**2*(j11 - j22)**2*(emax3 - om_d)**2/j00**2 + emax1**2*(j00 - j22)**2*(emax3 - om_d)**2/j11**2 + emax2**2*emax1**2*(j00 - j11)**2/j22**2
        ]

        for ub in upper_bound_fb_lin:                       # For all constraint variants
            for corner in box_corners:                      # For all box corners
                for i in range(P_offsetfree.Nc):            # For all individual constraints
                    a_i = ca.DM(P_offsetfree.A[i,:])
                    b_i = P_offsetfree.b[i] - np.sign(P_offsetfree.b[i]) * ca.sqrt(ub)     # sqrt because the upper bounds is |f|^2 instead of |f|

                    constraints.append(
                        ca.norm_2(a_i[:3] * r_empc)
                        + a_i.T @ (
                            self.M @ f_virt
                            + ca.vertcat(ca.DM([0]*3), (-k_omega) @ ca.vertcat(*corner))
                        )
                        <= b_i
                    )

        opti.subject_to(constraints)

        obj = 0
        obj += 3 * ca.log(r_empc) * 5
        for i in range(3):
            obj += ca.log(2 * k_omega[i,i] * bound[i])

        opti.minimize(-obj)

        opti.solver('ipopt', {}, {"print_level": 0})
        opti.set_initial(bound, 0.01* np.ones(bound.shape))
        # opti.set_initial(bound, 0.08* np.ones(bound.shape))
        # opti.set_initial(bound, np.array([0.4762, 0.54272, 0.81816, 0.08929]))
        try:
            sol = opti.solve()
        except:
            # 1. Extract current variable values
            x_current = opti.debug.value(bound)
            print(f"Current x values: {x_current}")
            
            # 2. Check constraint violations
            for i, constraint in enumerate(constraints):
                violation = opti.debug.value(constraint)
                print(f"Constraint {i} violation: {violation}")
            
            # 3. Check objective value
            obj_value = opti.debug.value(obj)
            print(f"Current objective value: {obj_value}")
        
        print(sol.value(bound))

        return sol.value(bound)

    def calc_empc(self, uimax, horizon, time_scaling = 5):
        # Continuous-time dynamics
        A = np.array([[0,1],[0,0]])
        B = np.array([[0, 1]]).T

        print(f"uimax: {uimax}")

        # Discrete-time dynamics
        sys = LinearSystem.from_continuous(A, B, time_scaling*self.model.dt, 'zero_order_hold')


        # # looks wrong
        # if self.R[0,0] != self.R[1,1]:
        #     raise ValueError("With the chosen Qu_bar, the first two elements of R need to be equal.")
        # # here find the correct R
        # R = np.array(self.R[0,0])**2 * self.r**2 * self.mass**4 + np.array(self.R[0,0]) * self.mass**2
        # if self.Q[0,0] != self.Q[1,1] or self.Q[2,2] != self.Q[3,3]:
        #     raise ValueError("Not mathematically necessary, but the given implementation " + 
        #                      "requires Q[0,0] = Q[1,1] and Q[2,2] = Q[3,3].")
        #     # otherwise, the empc controller needs to be calculated twice with the different parameters
        # Q = np.array([[self.Q[0,0], 0],[0, self.Q[2,2]]])

        # max eig value of R
        R = max(np.linalg.eigvals( self.Qu_tilde[0:3,0:3] ))
        R = np.array([[R]])

        if self.Q[0,0] != self.Q[1,1] or self.Q[0,0] != self.Q[2,2] or \
                self.Q[3,3] != self.Q[4,4] or self.Q[3,3] != self.Q[5,5]:
            raise ValueError("With given implementation all Q[pos] and all Q[vel] should be the same")
        Q = np.array([[self.Q[0,0], 0],[0, self.Q[3,3]]])

        # Adapt Q and R to the time scaling
        R *= time_scaling
        Q *= time_scaling
        
        # Terminal controller for the explicit MPC
        P, K = sys.solve_dare(Q, R)

        # State and input constraints
        U = Polyhedron.from_bounds(-uimax, uimax)
        X = Polyhedron.from_bounds(np.array([-5, -1.5]), # TODO parametrize
                                   np.array([ 5,  1.5]))
        D = X.cartesian_product(U)

        # Terminal set for the explicit MPC
        X_N = sys.mcais(K, D)

        # Calculate controller
        controller = ModelPredictiveController(sys, horizon, Q, R, P, D, X_N)
        controller.store_explicit_solution(verbose=True)

        return controller

    def bound_empc_cost(self, empc):
        # Get the vertices of all controlled sets
        allvertices = []
        for active_set in empc.explicit_solution.critical_regions:
            allvertices.extend(active_set.polyhedron.vertices)

        covered_area = MyPolytope.from_vertices(allvertices)
        # allvertices = np.array(allvertices).T

        # reduce the calculation effort # TODO parametrize this somehow
        bounds_min = np.array([-5,-5])
        bounds_max = np.array([ 5, 5])

        # Get tuples of the bounds and step size in each dimension
        slices = [slice(start, stop, 0.1) for start, stop in zip(bounds_min, bounds_max)]

        # Get all points in the space
        points = np.mgrid[slices].reshape(2,-1).T

        opti = ca.Opti()

        a = opti.variable(empc.S.nx, empc.S.nx, 'symmetric') # A needs to be positive (semi)-definite
        b = opti.variable(empc.S.nx)
        c = opti.variable(1)

        conditions = []
        obj = 0

        for point in points:
            val = empc.explicit_solution.V(point)
            if val is None:
                continue
            val = ca.DM(val)
            point = ca.DM(point)
            opt = point.T @ a @ point + b.T @ point + c

            conditions.append(val <= opt)
            obj += (opt-val)**2

        # conditions.append( c <= 0.05 )

        opti.subject_to(conditions)
        opti.minimize(obj)
        opti.solver('ipopt')

        # opti.set_initial(a, controller.explicit_solution.critical_regions[0]._V['xx'])
        opti.set_initial(b, empc.explicit_solution.critical_regions[0]._V['x'])
        opti.set_initial(c, 0)

        sol = opti.solve()
        terminal_cost = {'xx': sol.value(a), 'x': sol.value(b), 'c': sol.value(c)}
        print(f"xx: {terminal_cost['xx']}, x: {terminal_cost['x']}, c: {terminal_cost['c']}")
        return terminal_cost, covered_area

    def calculate_terminal_ingredients(self):
        # Calculate eMPC input bounds
        empc_bound = self.calc_empc_input_bounds()
        self.empc_bound_omega = empc_bound[:3]
        self.empc_bound_r = empc_bound[3]

        uimax = 1/np.sqrt(3) * self.empc_bound_r

        # Calculate eMPC and cost
        empc_horizon = self.tuning["empc_horizon"]
        time_scaling = self.tuning["time_scaling"]
        empc = self.calc_empc(uimax, empc_horizon, time_scaling)

        # Calculate bound and terminal set for each layer
        t_cost, t_set = self.bound_empc_cost(empc)

        # Calculate the complete cost
        ep1, ep2, ep3, ev1, ev2, ev3, eo1, eo2, eo3 = sp.symbols('ep1, ep2, ep3, ev1, ev2, ev3, eo1, eo2, eo3')

        # cost empc
        err_x = sp.Matrix([ep1, ev1])
        err_y = sp.Matrix([ep2, ev2])
        err_z = sp.Matrix([ep3, ev3])
        err_omega = sp.Matrix([eo1, eo2, eo3])

        cost_empc  = err_x.T @ sp.Matrix(t_cost['xx']) @ err_x + sp.Matrix(t_cost['x']).T @ err_x #+ sp.Matrix([[costs[0]['c']])
        cost_empc += err_y.T @ sp.Matrix(t_cost['xx']) @ err_y + sp.Matrix(t_cost['x']).T @ err_y #+ sp.Matrix([[costs[0]['c']])
        cost_empc += err_z.T @ sp.Matrix(t_cost['xx']) @ err_z + sp.Matrix(t_cost['x']).T @ err_z #+ sp.Matrix([[costs[0]['c']])
        cost_empc = cost_empc[0, 0] # make scalar

        # cost omega
        Qu_tilde_abs = np.linalg.norm(self.Qu_tilde)

        A_omega_subsystem = np.eye(3) - self.k_omega * self.dt
        Q_omega_subsystem = self.Q[6:9, 6:9] + 2 * Qu_tilde_abs * self.k_omega.T @ self.k_omega
        P_omega_subsystem = la.solve_discrete_lyapunov(A_omega_subsystem, Q_omega_subsystem)

        cost_omega = err_omega.T @ P_omega_subsystem @ err_omega
        cost_omega = cost_omega[0, 0] # make scalar

        # cost cross term 1
        rNorm = np.linalg.norm(self.r)
        omd = np.linalg.norm(self.omega_des)
        m = self.mass
        j00 = self.J[0,0]
        j11 = self.J[1,1]
        j22 = self.J[2,2]

        qu1 = self.Q[0,0]
        qu2 = self.Q[1,1]
        qu3 = self.Q[2,2]
        qu4 = self.Q[3,3]
        qu5 = self.Q[4,4]
        qu6 = self.Q[5,5]
        k1 = self.k_omega[0,0]
        k2 = self.k_omega[1,1]
        k3 = self.k_omega[2,2]

        # cross_1 = 2 * Qu_tilde_abs * \
        #     (rNorm**2*eo1**4 + ((-2*j00*j11/j22**2 + rNorm**2*j00**2/j22**2 - 2*rNorm**2*j00/j22 + rNorm**2*j11**2/j22**2 + 2*rNorm**2*j11/j22 + rNorm**2 + j00**2/j22**2 + j11**2/j22**2 - 2*rNorm**2*j00*j11/j22**2)*eo2**2 + (-2*j00*j22/j11**2 + 2*rNorm**2 + j00**2/j11**2 + j22**2/j11**2)*eo3**2 + (-4*j00*j22*omd/j11**2 + 2*j00**2*omd/j11**2 + 2*j22**2*omd/j11**2 + 4*rNorm**2*omd)*eo3 - 2*j00*j22*omd**2/j11**2 + j00**2*omd**2/j11**2 + j22**2*omd**2/j11**2)*eo1**2 + ((rNorm**2 - 2*rNorm**2*j11*j22/j00**2 + j11**2/j00**2 + j22**2/j00**2 + 2*rNorm**2*j11/j00 - 2*rNorm**2*j22/j00 + rNorm**2*j11**2/j00**2 + rNorm**2*j22**2/j00**2 - 2*j11*j22/j00**2)*eo3**2 + (-4*rNorm**2*j11*j22*omd/j00**2 + 4*rNorm**2*j11*omd/j00 - 4*rNorm**2*j22*omd/j00 + 2*rNorm**2*j11**2*omd/j00**2 + 2*rNorm**2*j22**2*omd/j00**2 - 4*j11*j22*omd/j00**2 + 2*rNorm**2*omd + 2*j11**2*omd/j00**2 + 2*j22**2*omd/j00**2)*eo3 + rNorm**2*omd**2 - 2*rNorm**2*j11*j22*omd**2/j00**2 + j11**2*omd**2/j00**2 + j22**2*omd**2/j00**2 + 2*rNorm**2*j11*omd**2/j00 - 2*rNorm**2*j22*omd**2/j00 + rNorm**2*j11**2*omd**2/j00**2 + rNorm**2*j22**2*omd**2/j00**2 - 2*j11*j22*omd**2/j00**2)*eo2**2 + rNorm**2*eo3**4 + 4*rNorm**2*eo3**3*omd + 4*rNorm**2*eo3**2*omd**2)

        # # cost cross term 2
        # input_empc_max = self.empc_bound_r
        # cross_2 = 2 * input_empc_max * \
        #     (sp.sqrt(m**4*qu2**2*rNorm**2*eo1**4 + ((m**4*qu1**2*rNorm**4 + 2*j00*j22*m**2*qu1*qu6*rNorm**2 - 2*j11*j22*m**2*qu1*qu6*rNorm**2 + m**4*qu1**2*rNorm**2 + j00**2*j22**2*qu6**2 - 2*j00*j11*j22**2*qu6**2 + j11**2*j22**2*qu6**2)*eo2**2 + (2*m**4*qu2**2*rNorm**2 + j00**2*j11**2*qu5**2 - 2*j00*j11**2*j22*qu5**2 + j11**2*j22**2*qu5**2)*eo3**2 + (4*m**4*qu2**2*rNorm**2*omd + 2*j00**2*j11**2*qu5**2*omd - 4*j00*j11**2*j22*qu5**2*omd + 2*j11**2*j22**2*qu5**2*omd)*eo3 + k1**2*m**4*qu3**2*rNorm**4 + 2*j00**2*k1**2*m**2*qu3*qu4*rNorm**2 + k1**2*m**4*qu3**2*rNorm**2 + j00**4*k1**2*qu4**2 + j00**2*j11**2*qu5**2*omd**2 - 2*j00*j11**2*j22*qu5**2*omd**2 + j11**2*j22**2*qu5**2*omd**2)*eo1**2 + ((-2*k1*m**4*qu3**2*rNorm**4 + 2*k3*m**4*qu1**2*rNorm**4 - 2*j00**2*k1*m**2*qu3*qu4*rNorm**2 + 2*j00*j11*k1*m**2*qu3*qu4*rNorm**2 - 2*j00*j22*k1*m**2*qu3*qu4*rNorm**2 + 2*j00*j22*k3*m**2*qu1*qu6*rNorm**2 - 2*j11*j22*k3*m**2*qu1*qu6*rNorm**2 + 2*j22**2*k3*m**2*qu1*qu6*rNorm**2 - 2*k1*m**4*qu3**2*rNorm**2 + 2*k3*m**4*qu1**2*rNorm**2 + 2*j00**3*j11*k1*qu4**2 - 2*j00**3*j22*k1*qu4**2 - 2*j00*j11**3*k2*qu5**2 + 2*j00*j22**3*k3*qu6**2 + 2*j11**3*j22*k2*qu5**2 - 2*j11*j22**3*k3*qu6**2)*eo3 - 2*k1*m**4*qu3**2*rNorm**4*omd - 2*j00**2*k1*m**2*qu3*qu4*rNorm**2*omd + 2*j00*j11*k1*m**2*qu3*qu4*rNorm**2*omd - 2*j00*j22*k1*m**2*qu3*qu4*rNorm**2*omd - 2*k1*m**4*qu3**2*rNorm**2*omd + 2*j00**3*j11*k1*qu4**2*omd - 2*j00**3*j22*k1*qu4**2*omd - 2*j00*j11**3*k2*qu5**2*omd + 2*j11**3*j22*k2*qu5**2*omd)*eo2*eo1 + ((m**4*qu3**2*rNorm**4 - 2*j00*j11*m**2*qu3*qu4*rNorm**2 + 2*j00*j22*m**2*qu3*qu4*rNorm**2 + m**4*qu3**2*rNorm**2 + j00**2*j11**2*qu4**2 - 2*j00**2*j11*j22*qu4**2 + j00**2*j22**2*qu4**2)*eo3**2 + (2*m**4*qu3**2*rNorm**4*omd - 4*j00*j11*m**2*qu3*qu4*rNorm**2*omd + 4*j00*j22*m**2*qu3*qu4*rNorm**2*omd + 2*m**4*qu3**2*rNorm**2*omd + 2*j00**2*j11**2*qu4**2*omd - 4*j00**2*j11*j22*qu4**2*omd + 2*j00**2*j22**2*qu4**2*omd)*eo3 + m**4*qu3**2*rNorm**4*omd**2 - 2*j00*j11*m**2*qu3*qu4*rNorm**2*omd**2 + 2*j00*j22*m**2*qu3*qu4*rNorm**2*omd**2 + m**4*qu3**2*rNorm**2*omd**2 + j00**2*j11**2*qu4**2*omd**2 - 2*j00**2*j11*j22*qu4**2*omd**2 + j00**2*j22**2*qu4**2*omd**2 + j11**4*k2**2*qu5**2)*eo2**2 + m**4*qu2**2*rNorm**2*eo3**4 + 4*m**4*qu2**2*rNorm**2*eo3**3*omd + (k3**2*m**4*qu1**2*rNorm**4 + 2*j22**2*k3**2*m**2*qu1*qu6*rNorm**2 + k3**2*m**4*qu1**2*rNorm**2 + 4*m**4*qu2**2*rNorm**2*omd**2 + j22**4*k3**2*qu6**2)*eo3**2))

        e_om_1 = eo1
        e_om_2 = eo2
        e_om_3 = eo3

        cross_1 = 2 * Qu_tilde_abs * \
            (rNorm**2*(e_om_1**4 / ( 1 - (1 - k1)**4 ) ) + ((-2*j00*j11/j22**2 + rNorm**2*j00**2/j22**2 - 2*rNorm**2*j00/j22 + rNorm**2*j11**2/j22**2 + 2*rNorm**2*j11/j22 + rNorm**2 + j00**2/j22**2 + j11**2/j22**2 - 2*rNorm**2*j00*j11/j22**2)*(e_om_2**2 / ( 1 - (1 - k2)**2 ) ) + (-2*j00*j22/j11**2 + 2*rNorm**2 + j00**2/j11**2 + j22**2/j11**2)*(e_om_3**2 / ( 1 - (1 - k3)**2 ) ) + (-4*j00*j22*omd/j11**2 + 2*j00**2*omd/j11**2 + 2*j22**2*omd/j11**2 + 4*rNorm**2*omd)*eo3 - 2*j00*j22*omd**2/j11**2 + j00**2*omd**2/j11**2 + j22**2*omd**2/j11**2)*(e_om_1**2 / ( 1 - (1 - k1)**2 ) ) + ((rNorm**2 - 2*rNorm**2*j11*j22/j00**2 + j11**2/j00**2 + j22**2/j00**2 + 2*rNorm**2*j11/j00 - 2*rNorm**2*j22/j00 + rNorm**2*j11**2/j00**2 + rNorm**2*j22**2/j00**2 - 2*j11*j22/j00**2)*(e_om_3**2 / ( 1 - (1 - k3)**2 ) ) + (-4*rNorm**2*j11*j22*omd/j00**2 + 4*rNorm**2*j11*omd/j00 - 4*rNorm**2*j22*omd/j00 + 2*rNorm**2*j11**2*omd/j00**2 + 2*rNorm**2*j22**2*omd/j00**2 - 4*j11*j22*omd/j00**2 + 2*rNorm**2*omd + 2*j11**2*omd/j00**2 + 2*j22**2*omd/j00**2)*eo3 + rNorm**2*omd**2 - 2*rNorm**2*j11*j22*omd**2/j00**2 + j11**2*omd**2/j00**2 + j22**2*omd**2/j00**2 + 2*rNorm**2*j11*omd**2/j00 - 2*rNorm**2*j22*omd**2/j00 + rNorm**2*j11**2*omd**2/j00**2 + rNorm**2*j22**2*omd**2/j00**2 - 2*j11*j22*omd**2/j00**2)*(e_om_2**2 / ( 1 - (1 - k2)**2 ) ) + rNorm**2*(e_om_3**4 / ( 1 - (1 - k3)**4 ) ) + 4*rNorm**2*(e_om_3**3 / ( 1 - (1 - k1)**3 ) )*omd + 4*rNorm**2*(e_om_3**2 / ( 1 - (1 - k3)**2 ) )*omd**2)

        # cost cross term 2
        input_empc_max = self.empc_bound_r
        cross_2 = 2 * input_empc_max * \
            (sp.sqrt(m**4*qu2**2*rNorm**2*(e_om_1**4 / ( 1 - (1 - k1)**4 ) ) + ((m**4*qu1**2*rNorm**4 + 2*j00*j22*m**2*qu1*qu6*rNorm**2 - 2*j11*j22*m**2*qu1*qu6*rNorm**2 + m**4*qu1**2*rNorm**2 + j00**2*j22**2*qu6**2 - 2*j00*j11*j22**2*qu6**2 + j11**2*j22**2*qu6**2)*(e_om_2**2 / ( 1 - (1 - k2)**2 ) ) + (2*m**4*qu2**2*rNorm**2 + j00**2*j11**2*qu5**2 - 2*j00*j11**2*j22*qu5**2 + j11**2*j22**2*qu5**2)*(e_om_3**2 / ( 1 - (1 - k3)**2 ) ) + (4*m**4*qu2**2*rNorm**2*omd + 2*j00**2*j11**2*qu5**2*omd - 4*j00*j11**2*j22*qu5**2*omd + 2*j11**2*j22**2*qu5**2*omd)*eo3 + k1**2*m**4*qu3**2*rNorm**4 + 2*j00**2*k1**2*m**2*qu3*qu4*rNorm**2 + k1**2*m**4*qu3**2*rNorm**2 + j00**4*k1**2*qu4**2 + j00**2*j11**2*qu5**2*omd**2 - 2*j00*j11**2*j22*qu5**2*omd**2 + j11**2*j22**2*qu5**2*omd**2)*(e_om_1**2 / ( 1 - (1 - k1)**2 ) ) + ((-2*k1*m**4*qu3**2*rNorm**4 + 2*k3*m**4*qu1**2*rNorm**4 - 2*j00**2*k1*m**2*qu3*qu4*rNorm**2 + 2*j00*j11*k1*m**2*qu3*qu4*rNorm**2 - 2*j00*j22*k1*m**2*qu3*qu4*rNorm**2 + 2*j00*j22*k3*m**2*qu1*qu6*rNorm**2 - 2*j11*j22*k3*m**2*qu1*qu6*rNorm**2 + 2*j22**2*k3*m**2*qu1*qu6*rNorm**2 - 2*k1*m**4*qu3**2*rNorm**2 + 2*k3*m**4*qu1**2*rNorm**2 + 2*j00**3*j11*k1*qu4**2 - 2*j00**3*j22*k1*qu4**2 - 2*j00*j11**3*k2*qu5**2 + 2*j00*j22**3*k3*qu6**2 + 2*j11**3*j22*k2*qu5**2 - 2*j11*j22**3*k3*qu6**2)*eo3 - 2*k1*m**4*qu3**2*rNorm**4*omd - 2*j00**2*k1*m**2*qu3*qu4*rNorm**2*omd + 2*j00*j11*k1*m**2*qu3*qu4*rNorm**2*omd - 2*j00*j22*k1*m**2*qu3*qu4*rNorm**2*omd - 2*k1*m**4*qu3**2*rNorm**2*omd + 2*j00**3*j11*k1*qu4**2*omd - 2*j00**3*j22*k1*qu4**2*omd - 2*j00*j11**3*k2*qu5**2*omd + 2*j11**3*j22*k2*qu5**2*omd)*eo2*eo1 + ((m**4*qu3**2*rNorm**4 - 2*j00*j11*m**2*qu3*qu4*rNorm**2 + 2*j00*j22*m**2*qu3*qu4*rNorm**2 + m**4*qu3**2*rNorm**2 + j00**2*j11**2*qu4**2 - 2*j00**2*j11*j22*qu4**2 + j00**2*j22**2*qu4**2)*(e_om_3**2 / ( 1 - (1 - k3)**2 ) ) + (2*m**4*qu3**2*rNorm**4*omd - 4*j00*j11*m**2*qu3*qu4*rNorm**2*omd + 4*j00*j22*m**2*qu3*qu4*rNorm**2*omd + 2*m**4*qu3**2*rNorm**2*omd + 2*j00**2*j11**2*qu4**2*omd - 4*j00**2*j11*j22*qu4**2*omd + 2*j00**2*j22**2*qu4**2*omd)*eo3 + m**4*qu3**2*rNorm**4*omd**2 - 2*j00*j11*m**2*qu3*qu4*rNorm**2*omd**2 + 2*j00*j22*m**2*qu3*qu4*rNorm**2*omd**2 + m**4*qu3**2*rNorm**2*omd**2 + j00**2*j11**2*qu4**2*omd**2 - 2*j00**2*j11*j22*qu4**2*omd**2 + j00**2*j22**2*qu4**2*omd**2 + j11**4*k2**2*qu5**2)*(e_om_2**2 / ( 1 - (1 - k2)**2 ) ) + m**4*qu2**2*rNorm**2*(e_om_3**4 / ( 1 - (1 - k3)**4 ) ) + 4*m**4*qu2**2*rNorm**2*(e_om_3**3 / ( 1 - (1 - k1)**3 ) )*omd + (k3**2*m**4*qu1**2*rNorm**4 + 2*j22**2*k3**2*m**2*qu1*qu6*rNorm**2 + k3**2*m**4*qu1**2*rNorm**2 + 4*m**4*qu2**2*rNorm**2*omd**2 + j22**4*k3**2*qu6**2)*(e_om_3**2 / ( 1 - (1 - k3)**2 ) )))

        self.terminal_cost = cost_empc + cost_omega + cross_1 + cross_2
        self.terminal_cost_function = sp.lambdify((ep1, ep2, ep3, ev1, ev2, ev3, eo1, eo2, eo3), self.terminal_cost)

        self.terminal_set = self.calc_terminal_set(t_set, self.empc_bound_omega)

    def create_python_code(self, cost):
        """
        Make the sympy code normal Python code and save it
        """
        python_code = sp.python(cost)

        for line in python_code.split("\n"):
            if line.startswith("e = "):
                final_expression = line.replace("e = ", "")
                break

        code = f"sp.lambdify((ep1, ep2, ep3, ev1, ev2, ev3, eo1, eo2, eo3), {final_expression}, modules={{'Abs':ca.fabs, 'tanh':ca.tanh}})"
        # execute code with cost = eval()
        return code

    def get_terminal_cost(self):
        if self.terminal_cost is None:
            raise ValueError("Terminal cost not calculated.")
        return self.create_python_code(self.terminal_cost)

    def calc_terminal_set(self, empcSet, ebound):

        empcZeros = np.zeros((empcSet.Nc, 1))
        empcA0 = empcSet.A[:,0].reshape(-1,1)
        empcA1 = empcSet.A[:,1].reshape(-1,1)

        A = np.block([
            [ empcA0,    empcZeros, empcZeros, empcA1   , empcZeros, empcZeros, np.zeros((empcSet.Nc, 3)) ],
            [ empcZeros, empcA0,    empcZeros, empcZeros, empcA1,    empcZeros, np.zeros((empcSet.Nc, 3)) ],
            [ empcZeros, empcZeros, empcA0,    empcZeros, empcZeros, empcA1,    np.zeros((empcSet.Nc, 3)) ],
            [ np.zeros((1, 6)), np.array([ 1, 0, 0]) ],
            [ np.zeros((1, 6)), np.array([-1, 0, 0]) ],
            [ np.zeros((1, 6)), np.array([ 0, 1, 0]) ],
            [ np.zeros((1, 6)), np.array([ 0,-1, 0]) ],
            [ np.zeros((1, 6)), np.array([ 0, 0, 1]) ],
            [ np.zeros((1, 6)), np.array([ 0, 0,-1]) ]
        ])

        b = np.vstack((
            empcSet.b.reshape(-1,1),
            empcSet.b.reshape(-1,1),
            empcSet.b.reshape(-1,1),
            ebound[0],
            ebound[0],
            ebound[1],
            ebound[1],
            ebound[2],
            ebound[2],
        ))

        return MyPolytope(A, b)

class PolytopeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MyPolytope):
            return {
                'A': obj.A.tolist(),
                'b': obj.b.tolist()
            }
        return super().default(obj)

class PolytopeDecoder(json.JSONDecoder):
    def decode(self, s):
        obj = json.loads(s)
        if 'A' in obj and 'b' in obj:
            return MyPolytope(np.array(obj['A']), np.array(obj['b']))
        return obj

def store_terminal_ingredients(cost_code, term_set, path):
    ingredients = {
        "cost": cost_code, 
        "term_set": json.dumps(term_set, cls=PolytopeEncoder)
    }
    yaml.dump(ingredients, open(path, "w"))

def load_terminal_ingredients(path):
    ingredients = yaml.safe_load(open(path))
    
    t_cost =eval(ingredients["cost"], {
        'sp':sp,
        'Symbol':sp.Symbol,
        'Float':sp.Float,
        'Abs':sp.Abs,
        'tanh':sp.tanh,
        'sqrt':sp.sqrt,
        'ca':ca,
        'ep1':sp.Symbol('ep1'),
        'ep2':sp.Symbol('ep2'),
        'ep3':sp.Symbol('ep3'),
        'ev1':sp.Symbol('ev1'),
        'ev2':sp.Symbol('ev2'),
        'ev3':sp.Symbol('ev3'),
        'eo1':sp.Symbol('eo1'),
        'eo2':sp.Symbol('eo2'),
        'eo3':sp.Symbol('eo3')
    }) 
    t_set = json.loads(ingredients["term_set"], cls=PolytopeDecoder)

    return t_cost, t_set

if __name__ == "__main__":
    import yaml

    from models.sys_model import SystemModel
    from models.spiral_model import SpiralModel
    from util.broken_thruster import BrokenThruster

    params = yaml.safe_load(open("config/reactive.yaml"))
    dt = params["time_step"]
    sim_duration = params["traj_duration"]

    # Initialize system model
    model = SystemModel(dt)
    for failure in params["actuator_failures"]:
        f = BrokenThruster(failure["act_id"], failure["intensity"])
        model.set_fault(f)

    # Initialize controller
    spiral_mpc_params = params["tuning"]["spiraling"]
    tuning = spiral_mpc_params[spiral_mpc_params["param_set"]]
    spiral_model = SpiralModel.from_system_model(model)

    terminal = explicitMPCTerminalIngredients(spiral_model, tuning)
    terminal.calculate_terminal_ingredients()

    code = terminal.get_terminal_cost()
    term_set = terminal.terminal_set
    store_terminal_ingredients(code, term_set, "terminal.yaml")


