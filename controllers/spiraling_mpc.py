import numpy as np
import casadi as ca

from util.controller_debug import ControllerDebug, DebugVal

class SpiralingController:
    """
    Controller implementing micro-orbiting
    """