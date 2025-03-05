<!-- markdownlint-disable -->

# <kbd>module</kbd> `sim_env`






---

## <kbd>class</kbd> `SimulationEnvironment`
Simulation environment for a 3D rigid body using Euler forward discretization. This class handles the dynamics of the rigid body and interfaces with a controller. 

### <kbd>method</kbd> `__init__`

```python
__init__(dt=0.01)
```

Initialize the simulation environment. 



**Args:**
 
 - <b>`dt`</b> (float):  Time step for simulation (seconds) 
 - <b>`mass`</b> (float):  Mass of the rigid body (kg) 
 - <b>`inertia`</b> (ndarray):  3x3 inertia tensor matrix (kg*m^2) 




---

### <kbd>method</kbd> `apply_external_forces`

```python
apply_external_forces(force=None, torque=None)
```

Apply external forces and torques to the rigid body. 



**Args:**
 
 - <b>`force`</b> (ndarray):  External force vector in world frame 
 - <b>`torque`</b> (ndarray):  External torque vector in body frame 

---

### <kbd>method</kbd> `compute_dynamics`

```python
compute_dynamics()
```

Compute the dynamics of the rigid body based on current state and applied forces. 



**Returns:**
 
 - <b>`tuple`</b>:  (linear_acceleration, angular_acceleration) 

---

### <kbd>method</kbd> `get_state`

```python
get_state()
```

Get the current state of the rigid body. 



**Returns:**
 
 - <b>`dict`</b>:  Current state of the rigid body 

---

### <kbd>method</kbd> `run_simulation`

```python
run_simulation(duration)
```

Run the simulation for a specified duration. 



**Args:**
 
 - <b>`duration`</b> (float):  Duration to run the simulation in seconds 

---

### <kbd>method</kbd> `set_controller`

```python
set_controller(controller)
```

Set the controller for the simulation. 



**Args:**
 
 - <b>`controller`</b>:  Controller object that implements a compute_control method 

---

### <kbd>method</kbd> `set_initial_state`

```python
set_initial_state(
    position=None,
    velocity=None,
    orientation=None,
    angular_velocity=None
)
```

Set the initial state of the rigid body. 



**Args:**
 
 - <b>`position`</b> (ndarray):  Initial position (x, y, z) 
 - <b>`velocity`</b> (ndarray):  Initial linear velocity 
 - <b>`orientation`</b>:  Initial orientation (as scipy Rotation or quaternion) 
 - <b>`angular_velocity`</b> (ndarray):  Initial angular velocity in body frame 

---

### <kbd>method</kbd> `step`

```python
step()
```

Step the simulation forward by one time step using Euler integration. If a controller is present, it will be called to compute control inputs. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
