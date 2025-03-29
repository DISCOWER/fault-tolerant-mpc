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
    q = q.squeeze()
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

def RotCasadi(q):
    """
    Rotation matrix from quaternion. Version for using ca.MX

    Args:
        q (ca.MX): Quaternion [x, y, z, w]
    
    Returns:
        ca.MX: 3x3 rotation matrix
    """
    R = ca.MX.zeros(3, 3)
    R[0, 0] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    R[0, 1] = 2*(q[0]*q[1] + q[2]*q[3])
    R[0, 2] = 2*(q[0]*q[2] - q[1]*q[3])

    R[1, 0] = 2*(q[0]*q[1] - q[2]*q[3])
    R[1, 1] = -q[0]**2 + q[1]**2 - q[2]**2 + q[3]**2
    R[1, 2] = 2*(q[1]*q[2] + q[0]*q[3])

    R[2, 0] = 2*(q[0]*q[2] + q[1]*q[3])
    R[2, 1] = 2*(q[1]*q[2] - q[0]*q[3])
    R[2, 2] = -q[0]**2 - q[1]**2 + q[2]**2 + q[3]**2
    return R

def RotFull(q):
    """
    Rotation of the full input vector

    Args:
        q (ndarray): Quaternion [x, y, z, w]
    
    Returns:
        ndarray: 6x6 rotation matrix
    """
    return np.block([
        [Rot(q), np.zeros((3,3))],
        [np.zeros((3,3)), np.eye(3)]
    ])

def RotFullInv(q):
    """ Inverse rotation of RotFull """
    return RotFull(q).T
