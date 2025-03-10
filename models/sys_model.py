import numpy as np
import casadi as ca

# from util.utils import Rot, RotInv
from util.broken_thruster import BrokenThruster
from util.utils import RotCasadi

def OmegaOperator(omega):
    """
    Omega operator for calculating Hamilton product of 3dim omega with a quaternion.

    Args:
        omega (ca.MX): 3x1 angular velocity vector
    
    Returns:
        ca.MX: 4x4 matrix
    """
    mat = ca.MX.zeros(4,4)

    mat[1,0] = -omega[2]
    mat[0,1] =  omega[2]
    mat[2,0] =  omega[1]
    mat[0,2] = -omega[1]
    mat[2,1] = -omega[0]
    mat[1,2] =  omega[0]
    mat[3,0:3] = -omega.T
    mat[0:3,3] = omega   

    return mat

class SystemModel:
    """
    System model for a 3D rigid body using Euler forward discretization.

    System state in form
        [ 
            pos         (3x1), global coords
            vel         (3x1), global coords
            orientation (4x1), quaternion
            ang_vel     (3x1), body coords
        ]
    """

    def __init__(self, dt):
        """
        Initialize system

        Args:
            dt (float): Time step for simulation (seconds)
        """
        # KTH Freeflyer-ish dynamics
        self.mass = 16.8
        self.inertia = np.array([
            [0.2, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.25]
        ])
        self.inertia_inv = np.linalg.inv(self.inertia)

        self.max_thrust = 3.4

        self.fun_options = {
            "jit": False,
            "jit_options": {"flags": ["-O2"]}
        }

        self.Nx = 13
        self.Nu_simplified = 6
        self.Nu_full = 16

        self.dt = dt

        self.D = np.zeros((self.Nu_simplified, self.Nu_full))
        d1 = 0.12 # distance from center to thruster
        d2 = 0.09

        # force, x-direction
        self.D[0, 0] = -1
        self.D[0, 1] = -1
        self.D[0, 2] =  1
        self.D[0, 3] =  1
        self.D[0, 4] = -1
        self.D[0, 5] = -1
        self.D[0, 6] =  1
        self.D[0, 7] =  1
        # force, y+direction
        self.D[1, 8] = -1
        self.D[1, 9] = -1
        self.D[1,10] =  1
        self.D[1,11] =  1
        # force, z+direction
        self.D[2,12] = -1
        self.D[2,13] =  1
        self.D[2,14] = -1
        self.D[2,15] =  1
        # torque, x-direction
        self.D[3,12] = -d1
        self.D[3,13] =  d1
        self.D[3,14] =  d1
        self.D[3,15] = -d1
        # torque, y-direction
        self.D[4, 0] = -d1
        self.D[4, 1] =  d1
        self.D[4, 2] =  d1
        self.D[4, 3] = -d1
        self.D[4, 4] = -d1
        self.D[4, 5] =  d1
        self.D[4, 6] =  d1
        self.D[4, 7] = -d1
        # torque, z-direction
        self.D[5, 0] =  d1
        self.D[5, 1] =  d1
        self.D[5, 2] = -d1
        self.D[5, 3] = -d1
        self.D[5, 4] = -d1
        self.D[5, 5] = -d1
        self.D[5, 6] =  d1
        self.D[5, 7] =  d1
        self.D[5, 8] = -d2
        self.D[5, 9] =  d2
        self.D[5,10] =  d2
        self.D[5,11] = -d2

        self.broken_thrusters = []
        self.faulty_force = np.zeros((1, self.Nu_full))
        self.faulty_force_generalized = self.D @ self.faulty_force.flatten()
        self.u_ub_physical = np.array([self.max_thrust] * self.Nu_full)

        self.set_dynamics()

    def set_dynamics(self):
        self.dynamics = self.rk4_integrator(self.dx_dt)

    def rk4_integrator(self, dynamics):
        """
        Runge-Kutta 4th Order discretization.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state at next step
        :rtype: ca.MX
        """
        x0 = ca.MX.sym('x0', self.Nx, 1)
        u = ca.MX.sym('u', self.Nu, 1)

        x = x0

        k1 = dynamics(x, u)
        k2 = dynamics(x + self.dt / 2 * k1, u)
        k3 = dynamics(x + self.dt / 2 * k2, u)
        k4 = dynamics(x + self.dt * k3, u)
        xdot = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        rk4 = ca.Function('RK4', [x0, u], [xdot], self.fun_options)

        return rk4

    def normalize_quaternion(self, state):
        """
        Normalize quaternion in the state to have unit length.

        Args:
            state (ca.MX): Full state vector
        
        Returns:
            state (ca.MX): State vector with normalized quaternion
        """
        state[6:10] = state[6:10] / ca.norm_2(state[6:10])
        return state

    def dx_dt(self, x, u):
        """
        Compute the dynamics of the rigid body based on current state and applied forces.

        Args:
            x (ca.MX): State vector
            u (ca.MX): Control vector

        Returns:
            xdot (ca.MX): Time derivative of the state vector
        """
        # Unpack state vector
        # position = x[0:3]
        velocity = x[3:6]
        orientation = x[6:10]
        angular_velocity = x[10:13]

        # Unpack control vector
        # for broken_thruster in self.broken_thrusters:
        #     u[broken_thruster.index] = 0.0

        u_elements = []
        for i in range(u.size1()):
            if any(i == bt.index for bt in self.broken_thrusters):
                u_elements.append(0.0)
            else:
                u_elements.append(u[i])
        
        # Create a new symbolic vector
        modified_u = ca.vertcat(*u_elements)

        generalised_force = ca.DM(self.D) @ (modified_u + ca.DM(self.faulty_force.reshape(-1, 1)))

        force = generalised_force[0:3]
        torque = generalised_force[3:6]

        # Compute linear acceleration (F = ma)
        dx_dt = velocity
        # Compute linear acceleration (F = ma)
        dv_dt = RotCasadi(orientation).T @ force / self.mass
        
        # Compute angular velocity in body frame
        dq_dt = 0.5 * OmegaOperator(angular_velocity) @ orientation
        # Compute angular acceleration in body frame
        # α = I^(-1) * (τ - ω×(I*ω))
        inertia_omega = self.inertia @ angular_velocity
        cross_term = ca.cross(angular_velocity, inertia_omega)
        domega_dt = self.inertia_inv @ (torque - cross_term)
        
        return ca.vertcat(*[dx_dt, dv_dt, dq_dt, domega_dt])

    def set_fault(self, broken_thruster):
        """
        Set thruster failure for simulation

        Args:
            broken_thruster (BrokenThruster): BrokenThruster object
        """
        self.broken_thrusters.append(broken_thruster)
        self.faulty_force = np.zeros(self.Nu)
        self.u_ub_physical = np.array([self.max_thrust] * self.Nu)
        for thruster in self.broken_thrusters:
            self.faulty_force[thruster.index] = thruster.intensity
            self.u_ub_physical[thruster.index] = 0.0
        self.faulty_force_generalized = self.D @ self.faulty_force.flatten()

        self.set_dynamics()

    @property
    def Nu(self):
        return self.Nu_full

    
if __name__ == "__main__":
    np.set_printoptions(linewidth=150)

    model = SystemModel()
    broken_thruster = BrokenThruster(0, 0.5)
    model.set_fault(broken_thruster)

    print(model.faulty_force)
    print(model.broken_thrusters)

    print(model.D)