import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from ft_mpc.simulation.sim_env import SimulationEnvironment
from ft_mpc.models.sys_model import SystemModel
from ft_mpc.models.spiral_model import SpiralModel
from ft_mpc.controllers.dummy_controller import Controller
from ft_mpc.controllers.spiraling_mpc import SpiralingController
from ft_mpc.util.broken_thruster import BrokenThruster
from ft_mpc.util.controller_debug import ControllerDebug

# Prepare parameters
params = yaml.safe_load(open(str(Path(__file__).absolute().parent) + "/../ft_mpc/config/reactive.yaml"))
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
    model.set_fault(f)

# Initialize controller
spiral_mpc_params = params["tuning"]["spiraling"]
spiral_model = SpiralModel.from_system_model(model)

# b = spiral_model.spiral_params.input_bounds.conv_hull
# from util.polytope import MyPolytope
# import matplotlib.pyplot as plt
# MyPolytope(b.A[:,0:3], b.b).plot()
# plt.show()

controller = SpiralingController(spiral_model, spiral_mpc_params, history)
# controller = Controller(model, history)

controller.load_trajectory(params["traj_shape"], sim_duration)

# Set up simulation environment
sim_env = SimulationEnvironment(model, controller)

sim_env.set_initial_state(
    position=[1,0,1],
    velocity=[1,0.5,0],
    orientation=R.from_euler("zyx", [50,30,-10], degrees=True).as_quat(),
    angular_velocity=[0.3,0.8,-0.1]
)

# Run simulation
sim_env.run_simulation(sim_duration)

# Evaluate results
# history.show_orbit_errors()
# history.show_generalized_inputs()
# history.show_direct_inputs()
history.export()
anim = history.animate_3d(spiral_model)
plt.show()

"""
Todo:
- [ ] write controller
    - [ ] spiral parameters
    - [x] input bounds -> replace with real implementation
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
