import numpy as np
import matplotlib.pyplot as plt
import yaml

from sim_env import SimulationEnvironment
from models.sys_model import SystemModel
from models.spiral_model import SpiralModel
from controllers.dummy_controller import Controller
from controllers.spiraling_mpc import SpiralingController
from util.broken_thruster import BrokenThruster
from util.controller_debug import ControllerDebug

# Prepare parameters
params = yaml.safe_load(open("config/reactive.yaml"))
dt = params["time_step"]
sim_duration = params["traj_duration"]

history = ControllerDebug()

# Initialize system model
model = SystemModel(dt)
for failure in params["actuator_failures"]:
    if failure["start_time"] != 0:
        print("WARNING: Actuator failures are not supported yet at times other than 0. Skipping.")
        continue
    f = BrokenThruster(failure["act_id"], failure["intensity"])

# Initialize controller
spiral_mpc_params = params["tuning"]["spiraling"]
controller = Controller(SpiralModel(model), spiral_mpc_params, history)

# Set up simulation environment
sim_env = SimulationEnvironment(model, controller)

sim_env.set_initial_state(
    position=[0,0,0],
    velocity=[0,0,0],
    orientation=[0,0,0,1],
    angular_velocity=[0,0,0]
)

# Run simulation
sim_env.run_simulation(sim_duration)

# Evaluate results
history.show_robot_errors()
anim = history.animate_3d()
plt.show()

"""
Todo:
- [ ] write controller
    - [ ] spiral parameters
    - [ ] input bounds -> replace with real implementation
    - [x] control allocator

    - [x] set cost functions
    - [x] build solver
    - [x] get_control

    - [ ] add fault

    - [ ] Fix the transform into the force-aligned system
        - [ ] Fix the rotations that exist in the controller
        - [ ] Think about how the inertia is then changed

    - [ ] implement the terminal cost and set
- [x] write spiral dynamics
- [x] transform robot->spiral and back
"""
