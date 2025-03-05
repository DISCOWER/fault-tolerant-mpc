import numpy as np
import casadi as ca

from util.controller_debug import ControllerDebug, DebugVal

class Controller:
    """
    Controller for the system.
    """

    def __init__(self, model, history):
        """
        Initialize the controller.

        Args:
            model (SystemModel): Model of the system.
        """
        self.model = model

        self.Nx = model.Nx
        self.Nu = model.Nu

        self.history = history

    def get_control(self, state, time):
        """
        Get the control input for the system.

        Args:
            state (ndarray): Current state of the system.

        Returns:
            ndarray: Control input for the system.
        """
        u = np.zeros(self.Nu)
        u[1] = 0.3

        debug = DebugVal(self, time)
        debug.set_state(state)
        debug.set_input(u, self.model)
        debug.set_desired_state(np.zeros(self.Nx))
        debug.calculate_errors()
        self.history.add_debug_val(debug)

        return u

