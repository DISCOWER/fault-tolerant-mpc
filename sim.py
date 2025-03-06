import numpy as np
import matplotlib.pyplot as plt

from sim_env import SimulationEnvironment
from model import SystemModel
from controllers.dummy_controller import Controller
from util.controller_debug import ControllerDebug

dt = 0.1
sim_duration = 10

history = ControllerDebug()

model = SystemModel(dt)
controller = Controller(model, history)

sim_env = SimulationEnvironment(model, controller)

sim_env.set_initial_state(
    position=[0,0,0],
    velocity=[0,0,0],
    orientation=[0,0,0,1],
    angular_velocity=[0,0,0]
)

sim_env.run_simulation(sim_duration)

history.show_robot_errors()
anim = history.animate_3d()
plt.show()

"""
Todo:
- [ ] write controller
"""
