import numpy as np
import casadi as ca

def Rot(q):
    """
    Rotation matrix from quaternion. 

    Args:
        q (ndarray): Quaternion [x, y, z, w]

    Returns:
        ndarray: 3x3 rotation matrix
    """
    return np.array([
        [q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2, 2*(q[0]*q[1] + q[2]*q[3]), 2*(q[0]*q[2] - q[1]*q[3])],
        [2*(q[0]*q[1] - q[2]*q[3]), -q[0]**2 + q[1]**2 - q[2]**2 + q[3]**2, 2*(q[1]*q[2] + q[0]*q[3])],
        [2*(q[0]*q[2] + q[1]*q[3]), 2*(q[1]*q[2] - q[0]*q[3]), -q[0]**2 - q[1]**2 + q[2]**2 + q[3]**2]
    ])

def RotInv(q):
    """
    Inverse rotation matrix from quaternion. 

    Args:
        q (ndarray): Quaternion [w, x, y, z]

    Returns:
        ndarray: 3x3 rotation matrix
    """
    return Rot(q).T

