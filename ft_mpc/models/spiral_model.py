import numpy as np
import casadi as ca

from ft_mpc.models.sys_model import SystemModel, OmegaOperator
from ft_mpc.controllers.tools.spiral_parameters import SpiralParameters
from ft_mpc.util.utils import RotCasadi, Rot, RotInv

class SpiralModel(SystemModel):
    """
    Spiraling model for micro-orbiting

    System state in form
        [
            pos         (3x1), global coords
            vel         (3x1), global coords
            ang_vel     (3x1), body coords
            orientation (4x1), quaternion
        ]
    where only the first 9 states are used for control.
    """
    def __init__(self, dt, spiral_params):
        self.r = spiral_params.r
        # self.omega_des = spiral_params.omega_des
        # self.virt_force_abs = np.abs(spiral_params.virt_force)
        self.spiral_params = spiral_params

        super().__init__(dt)

        self.set_dynamics()

    @classmethod
    def from_system_model(cls, sys_model):
        """
        Create a new object from an existing SystemModel object.
        """
        spiral_params = SpiralParameters(sys_model)
        new_spiral_model = cls(sys_model.dt, spiral_params)

        for broken_thruster in sys_model.broken_thrusters:
            new_spiral_model.set_fault(broken_thruster)

        return new_spiral_model

    def dx_dt(self, c, u):
        """
        Dynamics of the center point of the orbit.

        Args:
            x (ca.MX): State of the center of the orbit
            u (ca.MX): Control input (generalized force - 6d)

        Returns:
            ca.MX: Time derivative of the state
        """
        vel = c[3:6]
        omega = c[6:9]
        q = c[9:13]

        dpos_dt = vel

        generalized_force = u + self.faulty_force_generalized.reshape(-1, 1)
        force = generalized_force[0:3]
        torque = generalized_force[3:6]

        inertia_omega = self.inertia @ omega
        cross_term = ca.cross(omega, inertia_omega)
        domega_dt = self.inertia_inv @ (torque - cross_term)

        dvel_dt = RotCasadi(q).T @ (
            force / self.mass
            + ca.cross(domega_dt, self.r)
            + ca.cross(omega, ca.cross(omega, self.r))
        )

        dalpha_dt = 0.5 * OmegaOperator(omega) @ q
        return ca.vertcat(dpos_dt, dvel_dt, domega_dt, dalpha_dt)

    def normalize_quaternion(self, state):
        """
        Normalize quaternion in the state to have unit length.

        Args:
            state (ca.MX): Full state vector (center-point state)

        Returns:
            state (ca.MX): State vector with normalized quaternion (center-point state)
        """
        state[9:13] = state[9:13] / ca.norm_2(state[9:13])
        return state

    def robot_to_center(self, x):
        """
        Transform the robot state to the center-point state.

        Args:
            x (ca.MX): Full state vector (robot state)

        Returns:
            ca.MX: Center-point state vector
        """
        x = x.flatten()

        omega = x[10:13]
        q = x[6:10]

        pos = x[0:3] + RotInv(q) @ self.r
        vel = x[3:6] + RotInv(q) @ np.cross(omega.flatten(), self.r)

        return np.concatenate((pos, vel, omega, q))

    def center_to_robot(self, c):
        """
        Transform the center-point state to the robot state.

        Args:
            c (ca.MX): Center-point state vector

        Returns:
            ca.MX: Full state vector (robot state)
        """
        x = x.flatten()

        omega = c[6:9]
        q = c[9:13]

        pos = c[0:3] - Rot(q) @ self.r
        vel = c[3:6] - Rot(q) @ np.cross(omega, self.r)

        return np.concatenate((pos, vel, q, omega))

    @property
    def Nu(self):
        return self.Nu_simplified

"""
For later reference:
Transformation of an inertia tensor to another coordinate system:
    I' = R I R^T
where R is the rotation matrix from the old to the new coordinate system.
"""
