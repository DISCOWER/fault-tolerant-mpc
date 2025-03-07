import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import copy

from util.animate import animate_trajectory

class DebugVal:
    def __init__(self, controller, t):
        self.controller = str(controller)
        self.faulty_force = copy.deepcopy(controller.model.faulty_force)

        self.time = t

        self.position = None
        self.velocity = None
        self.orientation = None
        self.angular_velocity = None

        self.input = None
        self.force = None
        self.torque = None

        self.circle_position = None
        self.circle_velocity = None
        self.circle_angular_velocity = None

        self.position_error = None
        self.velocity_error = None
        self.orientation_error = None
        self.angular_velocity_error = None

        self.circle_position_error = None
        self.circle_velocity_error = None
        self.circle_angular_velocity_error = None

    def set_state(self, x):
        x = np.array(x).flatten()
        self.position = x[0:3]
        self.velocity = x[3:6]
        self.orientation = x[6:10]
        self.angular_velocity = x[10:13]

    def set_circle_state(self, c):
        c = np.array(c).flatten()
        self.circle_position = c[0:3]
        self.circle_velocity = c[3:6]
        self.circle_angular_velocity = c[6:9]

    def set_input(self, u, model):
        u = np.array(u).flatten()
        self.input = u
        generalized_force = model.D @ u
        self.force = generalized_force[0:3]
        self.torque = generalized_force[3:6]

    def set_desired_state(self, x):
        x = np.array(x).flatten()
        self.desired_position = x[0:3]
        self.desired_velocity = x[3:6]
        if np.size(x) == 9:
            self.desired_angular_velocity = x[6:9]
            self.desired_orientation = np.zeros(4)
        else:
            self.desired_orientation = x[6:10]
            self.desired_angular_velocity = x[10:13]

    def calculate_errors(self):
        if self.position is not None and self.desired_position is not None:
            self.position_error = self.desired_position - self.position
            self.velocity_error = self.desired_velocity - self.velocity
            self.orientation_error = self.desired_orientation - self.orientation
            self.angular_velocity_error = self.desired_angular_velocity - self.angular_velocity
        
        if self.circle_position is not None and self.desired_position is not None:
            self.circle_position_error = self.desired_position - self.circle_position
            self.circle_velocity_error = self.desired_velocity - self.circle_velocity
            self.circle_angular_velocity_error = self.desired_angular_velocity - self.circle_angular_velocity

class ControllerDebug:
    def __init__(self):
        self.history = []

    def add_debug_val(self, debug_val):
        self.history.append(debug_val)

    def get_time(self):
        # t = [h.time for h in self.history]
        # return np.array(t).T
        return [h.time for h in self.history]
    
    def show_direct_inputs(self):
        inputs = [h.input for h in self.history]
        inputs = np.array(inputs)

        time = self.get_time()

        fig, ax = plt.subplots(4, 4)
        positions = {
            0 : (0, 0),
            1 : (0, 1),
            2 : (0, 2),
            3 : (0, 3),
            4 : (1, 0),
            5 : (1, 1),
            6 : (1, 2),
            7 : (1, 3),
            8 : (2, 0),
            9 : (2, 1),
            10 : (2, 2),
            11 : (2, 3),
            12 : (3, 0),
            13 : (3, 1),
            14 : (3, 2),
            15 : (3, 3)
        }

        for i in range(16):
            ax[positions[i]].plot(time, inputs[:, i])
            ax[positions[i]].set_title(f"Input {i}")

        plt.tight_layout()

    def show_generalized_inputs(self):
        forces = [h.force for h in self.history]
        torques = [h.torque for h in self.history]

        forces = np.array(forces)
        torques = np.array(torques)

        time = self.get_time()

        fig, ax = plt.subplots(2, 3)

        for i in range(3):
            ax[0, i].plot(time, [f[i] for f in forces])
            ax[0, i].set_title(f"Force {i}")

            ax[1, i].plot(time, [t[i] for t in torques])
            ax[1, i].set_title(f"Torque {i}")

        plt.tight_layout()
    
    def show_robot_errors(self):
        position_errors = [h.position_error for h in self.history]
        velocity_errors = [h.velocity_error for h in self.history]
        orientation_errors = [h.orientation_error for h in self.history]
        angular_velocity_errors = [h.angular_velocity_error for h in self.history]

        position_errors = np.array(position_errors)
        velocity_errors = np.array(velocity_errors)
        orientation_errors = np.array(orientation_errors)
        angular_velocity_errors = np.array(angular_velocity_errors)

        time = self.get_time()

        fig, ax = plt.subplots(3, 2)
        for i in range(3):
            print(i)
            print(position_errors[:, i])

            ax[i, 0].plot(time, position_errors[:, i])
            ax[i, 0].set_title(f"Position Error {i}")

            ax[i, 1].plot(time, velocity_errors[:, i])
            ax[i, 1].set_title(f"Velocity Error {i}")

        plt.tight_layout()

        fig2, ax2 = plt.subplots(4, 2)
        for i in range(4):
            ax2[i, 0].plot(time, orientation_errors[:, i])
            ax2[i, 0].set_title(f"Orientation Error {i}")

        for i in range(3):
            ax2[i, 1].plot(time, angular_velocity_errors[:, i])
            ax2[i, 1].set_title(f"Angular Velocity Error {i}")

        plt.tight_layout()

    def animate_3d(self):
        position = [h.position for h in self.history]
        orientation = [h.orientation for h in self.history]
        input = [h.input for h in self.history]

        animate_trajectory(positions=position, 
                           quaternions=orientation, 
                           time=self.get_time(), 
                           inputs=input)

