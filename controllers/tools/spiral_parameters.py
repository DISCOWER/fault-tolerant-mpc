import numpy as np
from numpy.linalg import norm
import cvxpy as cp

from controllers.tools.input_bounds import InputBounds

class SpiralParameters:
    """
    Calculates the optimal parameters for the micro-orbit based on the current system failure state.
    """
    def __init__(self, model):
        self.model = model
        self.mass = model.mass
        self.inertia = model.inertia

        self.faulty_force = model.faulty_force.flatten()
        self.faulty_force_generalized = model.faulty_force_generalized.flatten()
        self.D = model.D

        self.input_bounds = InputBounds(model)

        self.calculate_optimal_parameters()

    def calculate_optimal_parameters(self):
        """
        Calculate the optimal parameters for the micro-orbit based on the current system failure state.
        """
        # "hard-code" the values for now
        self.omega_des = np.array([0.0, 0.5, 0.0])
        r_dir = np.array([0.0, 1.0, 0.0])

        self.f_virt = 1.5  * r_dir

        self.r = norm(self.f_virt) / (self.mass * norm(self.omega_des)**2) * r_dir

    def __str__(self):
        return f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"+ "\n" \
        + f"Spiral parameters - Overview over errors and constants:" + "\n" \
        + f"Generalized error: {self.faulty_force_generalized.T}" + "\n" \
        + f"Physical error: {self.faulty_force.T}" + "\n" \
        + f"Virtual error b: {self.f_virt.T} with norm {norm(self.f_virt)}" + "\n" \
        + f"Spiral constants: Radius: {self.r}, omega_theta: {self.omega_des}" + "\n" \
        + f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" + "\n" \

if __name__ == "__main__":
    from models.sys_model import SystemModel
    from util.broken_thruster import BrokenThruster
    model = SystemModel(dt=0.1)

    model.set_fault(BrokenThruster(0, 0.5))
    model.set_fault(BrokenThruster(1, 0.5))

    spiral_params = SpiralParameters(model)
    print(spiral_params)

