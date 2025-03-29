import warnings
import yaml
import numpy as np
from scipy.spatial.transform import Rotation

def euler_traj_to_quat_traj(euler_traj):
    """
    Convert a trajectory from euler angles to quaternions.

    :param euler_traj: trajectory in euler angles
    :type euler_traj: ndarray
    :return: trajectory in quaternions
    :rtype: ndarray
    """
    quat_traj = np.zeros((4, euler_traj.shape[1]))
    for i in range(euler_traj.shape[1]):
        r = Rotation.from_euler('xyz', euler_traj[:, i])
        quat_traj[:, i] = r.as_quat()

    return quat_traj

def quat_traj_to_angular_vel(quat_traj, dt):
    """
    Calculate the angular velocity from a trajectory of quaternions.

    :param quat_traj: trajectory in quaternions
    :type quat_traj: ndarray
    :param dt: time step
    :type dt: float
    :return: angular velocity
    :rtype: ndarray
    """
    quat_seq = [Rotation.from_quat(quat) for quat in quat_traj.T]
    angular_traj = np.zeros((3, quat_traj.shape[1]))

    for i in range(1, quat_traj.shape[1]):
        r_diff = quat_seq[i-1].inv() * quat_seq[i]
        rotvec = r_diff.as_rotvec()
        angular_traj[:, i] = rotvec / dt

    return angular_traj

def load_trajectory(action, dt, duration=100, file_path=None):
    """
    Load a trajectory for the controller to track.

    :param action: action to perform, either 'generate', 'generate_line' or 'load'
    :type action: str
    :param duration: duration of the trajectory, defaults to 'None'. This means the whole
                    trajectory should be loaded independent of the duration.
    :type duration: int, optional
    :param file_path: path to the trajectory file, defaults to None
    :type file_path: str, optional
    """
    def load_trajectory_from_file(file_path, dt):
        """
        Load a trajectory from a yaml file.

        :param file_path: path to the file
        :type file_path: str
        """
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        if data['dt'] != dt:
            raise ValueError(f"Trajectory ({data['dt']}s) and controller ({dt}s) " + 
                            "have different time steps.")

        return np.array(data['x']).T

    def generate_trajectory(duration, dt, form, **kwargs):
        """
        Get a trajectory for the robot to follow.
        """
        t = np.arange(0, 10*duration, dt ).reshape(1, -1)
        match form:
            case 'sin':
                quat_traj = euler_traj_to_quat_traj(np.stack((
                    np.ones(t.shape)*np.pi/2,   # x rot
                    np.zeros(t.shape),          # y rot
                    np.zeros(t.shape)           # z rot
                )).reshape(3, -1))
                omega_traj = quat_traj_to_angular_vel(quat_traj, dt)
                gain = 0.1
                x_ref = np.concatenate((
                            gain*np.sin(t),     # x
                            t,                  # y
                            np.zeros(t.shape),  # z
                            gain*np.cos(t),     # vx
                            np.ones(t.shape),   # vy
                            np.zeros(t.shape),  # vz
                            quat_traj,
                            omega_traj
                        ))
                
            case 'line':
                x_ref = np.concatenate((
                            t,                  # pos
                            np.zeros(t.shape),  
                            np.zeros(t.shape), 
                            np.ones(t.shape),   # vel 
                            np.zeros(t.shape), 
                            np.zeros(t.shape),
                            euler_traj_to_quat_traj(np.zeros((3, t.size))),
                            np.zeros((3, t.size))
                        ))
            case 'point_stabilizing' | 'hover':
                if 'position' in kwargs:
                    x_pos = kwargs['position']
                else:
                    x_pos = [0, 0, 0]
                x_ref = np.concatenate((
                    x_pos[0] * np.ones(t.shape),
                    x_pos[1] * np.ones(t.shape),
                    x_pos[2] * np.ones(t.shape),
                    np.zeros(t.shape),
                    np.zeros(t.shape),
                    np.zeros(t.shape),
                    euler_traj_to_quat_traj(np.zeros((3, t.size))),
                    np.zeros((3, t.size))
                ))
            case 'circle':
                radius = kwargs['radius'] if 'radius' in kwargs else 2
                sPerRot = kwargs['sPerFullCircle'] if 'sPerFullCircle' in kwargs else 30
                # full turn every <x>s
                omega = 2*np.pi/sPerRot 
                x_ref = np.concatenate((
                    radius * np.cos(omega * t),          # pos
                    radius * np.sin(omega * t),
                    np.zeros_like(t),
                    -radius * omega * np.sin(omega * t), # vel
                    radius * omega * np.cos(omega * t),
                    np.zeros_like(t),
                    euler_traj_to_quat_traj(np.zeros((3, t.size))),
                    np.zeros((3, t.size))
                ))

                x_ref += np.array( [-radius] + [0]*12 ).reshape(-1, 1)
            case _:
                raise ValueError("Invalid form. Use 'sin' or 'line'.")
        return x_ref

    if action == "generate_sin":
        t = generate_trajectory(duration, dt, form='sin')
    elif action == "generate_line":
        t = generate_trajectory(duration, dt, form='line')
    elif action == "generate_point_stabilizing" or action == "hover":
        t = generate_trajectory(duration, dt, form='point_stabilizing')
    elif 'hover' in action:
        name, *params = action.split('_')
        if name != 'hover':
            raise ValueError(f"Invalid action '{action}'.")
        if len(params) != 3:
            raise ValueError(f"Invalid number of parameters for action '{action}'. Use 'hover'"
                                +" or 'hover_<x>_<y>_<alpha>'")
        position = [float(p) for p in params]
        t = generate_trajectory(duration, dt, form='point_stabilizing', position=position)
    elif action == "generate_circle":
        t = generate_trajectory(duration, dt, form='circle')
    elif 'circle' in action:
        name, *params = action.split('_')
        if name != 'circle':
            raise ValueError(f"Invalid action '{action}'.")
        if len(params) != 4:
            raise ValueError(f"Invalid number of parameters for action '{action}'. Use 'circle_r_<radius>_sPerFullCircle_<speed>'")
        if params[0] != 'r' or params[2] != 'sPerFullCircle':
            raise ValueError(f"Invalid parameters for action '{action}'. Use 'circle_r_<radius>_sPerFullCircle<speed>'")
        radius = float(params[1])
        speed = float(params[3])
        t = generate_trajectory(duration, dt, form='circle', radius=radius, sPerFullCircle=speed)
    elif action == "load":
        if file_path is None:
            raise ValueError(f"Invalid parameters for action '{action}'. Use with 'file_path=<path>'.")

        t = load_trajectory_from_file(file_path, dt)

        if duration is not None:
            if t.shape[1] < (duration)/dt:
                raise ValueError(f"Warning: Trajectory is too short: Has only lenth of  " +
                                    f"{t.shape[1]*dt}s, but {duration}s needed.")
    else:
        raise ValueError(f"Invalid action '{action}'.")

    return t

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dt = 0.1
    duration = 100
    for cmd in ['generate_sin', 'generate_line', 'generate_point_stabilizing', 'hover_1_2_3', 'generate_circle', 'circle_r_2_sPerFullCircle_30']:
        t = load_trajectory(cmd, dt, duration=duration)

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot3D(t[0, :], t[1, :], t[2, :], 'gray')
        plt.show()