import numpy as np
import cvxpy as cp
from qpsolvers import solve_qp
 

class ControlAllocator:
    """
    Control allocator for the thrusters. Calculates the 16d thruster input from the 6d generalized 
    force.
    """

    def __init__(self, model, bounds):
        """
        Initialize control allocator

        Args:
            model (SpiralModel): System model
            bounds (InputBounds): Control bounds
        """
        self.model = model
        self.faulty_force_generalized = model.faulty_force_generalized.flatten()
        self.input_bounds = bounds

        N_u_full = 16
        N_u_simple = 6

        # Define CVXPY problem components
        self.u_desired = cp.Parameter(N_u_simple)
        self.upper_bound = cp.Parameter(N_u_full)
        self.u_phys_min_energy = cp.Variable(N_u_full)

        obj = cp.Minimize(cp.sum_squares(self.u_phys_min_energy))

        constraints = [
            self.u_phys_min_energy >= np.zeros(N_u_full),
            self.u_phys_min_energy <= self.upper_bound,
            self.model.D @ self.u_phys_min_energy == self.u_desired
        ]

        self.prob = cp.Problem(obj, constraints)

    def clip_generalized_input(self, u):
        """
        Clip the generalized input to the physical limits. Solve by optimizing the problem

            min(u_clipped) (u - u_clipped)^2
            s.t.           A u_clipped <= b

        using a QP solver.

        Args:
            u (np.array): Generalized input

        Returns:
            np.array: Clipped generalized input
        """
        A, b = self.input_bounds.get_conv_hull()

        if np.all(A @ u <= b):
            # constraints satisfied
            return u

        return solve_qp(np.identity(3), -u, A, b, solver="daqp")

    def get_physical_input(self, u_simple):
        """
        Get the physical (8-dim) input from the simplified (3-dim) input. The input is clipped to 
        the closest feasible solution if it is outside of the physical bounds.

        Args:
            u_simple (np.array): Simplified input

        Returns:
            np.array: Generalized input
        """
        u_des = u_simple.flatten()
        u_fault = self.faulty_force_generalized 

        u_des = self.clip_generalized_input(u_des + u_fault) - u_fault

        # Update parameter values
        self.u_desired.value = u_des
        self.upper_bound.value = self.model.u_ub_physical.flatten()

        # Solve the problem
        self.prob.solve()

        if self.prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Problem status: {self.prob.status}")
            print(f"u_des: {self.u_desired.value}")
            print(f"u_ub: {self.upper_bound.value}")
            exit()
            return None

        return self.u_phys_min_energy.value