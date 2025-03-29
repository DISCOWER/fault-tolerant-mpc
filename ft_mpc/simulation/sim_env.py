import numpy as np
from scipy.spatial.transform import Rotation


class SimulationEnvironment:
    """
    Simulation environment for a 3D rigid body using Euler forward discretization.
    This class handles the dynamics of the rigid body and interfaces with a controller.
    """

    def __init__(self, model, controller):
        """
        Initialize the simulation environment.
        
        Args:
            model (SystemModel): Model of the system.
            controller (Controller): Controller for the system.
            dt (float): Time step for simulation (seconds)
        """
        self.model = model
        self.set_controller(controller)
        self.dt = self.model.dt
        
        # e.g. position: max. rate of divergence is 1cm/s with dt=0.1s
        self.noise = {
            'position': 0.001,         # uniform distribution, unit: meter
            'velocity': 0.001,         # uniform distribution, unit: meter/second
            'orientation': 0.001,      # uniform distribution 
            'angular_velocity': 0.001, # uniform distribution, unit: rad/second
        }

        self.state = np.zeros(model.Nx)

        self.cur_time = 0.0

    def set_controller(self, controller):
        """
        Set the controller for the simulation.
        
        Args:
            controller (Controller): Controller for the system.
        """
        self.controller = controller

    def set_initial_state(self, position=None, velocity=None, orientation=None, angular_velocity=None):
        """
        Set the initial state of the rigid body.
        
        Args:
            position (ndarray): Initial position (x, y, z)
            velocity (ndarray): Initial linear velocity
            orientation (ndarray): Initial pos as quaternion [x, y, z, w]
            angular_velocity (ndarray): Initial angular velocity in body frame
        """
        if position is not None:
            self.state[0:3] = np.array(position)
        
        if velocity is not None:
            self.state[3:6] = np.array(velocity)
            
        if orientation is not None:
            self.state[6:10] = orientation
        
        if angular_velocity is not None:
            self.state[10:] = np.array(angular_velocity)

    def set_fault(self, fault):
        """
        Set thruster failure for simulation
        
        Args:
            fault (BrokenThruster): Thruster failure parameters
        """
        self.model.set_fault(fault)
        self.controller.set_fault(fault)

    def step(self):
        """
        Perform a single simulation step.
        """
        # Get control input from controller
        u = self.controller.get_control(self.state, self.cur_time)
        
        # Compute state derivatives
        x_new = self.model.dynamics(self.state, u)

        # add noise
        x_new[0:3]  += np.random.uniform(0, self.noise['position'], size=(3))
        x_new[3:6]  += np.random.uniform(0, self.noise['velocity'], size=(3))
        x_new[6:10] += np.random.uniform(0, self.noise['orientation'], size=(4))
        x_new[10:]  += np.random.uniform(0, self.noise['angular_velocity'], size=(3))

        x_new = self.model.normalize_quaternion(x_new)

        # Update state
        self.state = np.array(x_new)

        # Update time
        self.cur_time += self.dt
        

    def run_simulation(self, duration):
        """
        Run the simulation for a specified duration.
        
        Args:
            duration (float): Duration to run the simulation in seconds
        """
        n_steps = int(duration / self.dt)
        
        for step in range(n_steps):
            self.step()

