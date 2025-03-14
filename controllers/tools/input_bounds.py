import numpy as np
from itertools import product
from scipy.spatial import ConvexHull
import time

from util.polytope import MyPolytope

class InputBounds:
    """
    Calculates the bound on the generalized input based on individual thruster inputs.
    """
    def __init__(self, model):
        self.model = model
        self.conv_hull = None
        self.vertices = None
        self.calc_input_bounds()

    def get_conv_hull(self):
        """
        Gives back the convex hull of the possible resulting forces and torques that can be 
        applied. These are calculated based on the uncontrollable force of the model.

        :return: Matrix A and vector b of the convex hull
        :rtype: np.array, np.array
        """
        if self.conv_hull is None:
            self._calc_conv_hull()

        return self.conv_hull.A, self.conv_hull.b

    def get_vertices(self):
        """
        Get the vertices of the convex hull

        :return: Vertices of the convex hull
        :rtype: np.array
        """
        if self.vertices is None:
            self._calc_conv_hull()

        return self.vertices

    def calc_input_bounds(self):
        """
        Calculate the input bounds for the thrusters
        """
        # Get the min and max forces for each thruster
        min_max = []
        broken_idx = [thruster.index for thruster in self.model.broken_thrusters]
        f_fault = self.model.faulty_force.flatten()

        for i in range(self.model.Nu_full):
            if i in broken_idx:
                min_max.append([f_fault[i], f_fault[i]])
            else:
                min_max.append([0.0, self.model.max_thrust])

        list_of_input_forces = list(product(*min_max))

        # Calculate resulting generalized force (force+torque)
        vertices = [] 
        for input_force in list_of_input_forces:
            res_f = np.matmul(self.model.D, np.array(input_force))
            vertices.append(res_f)

        # Calculate the convex hull from the resulting forces
        self.vertices = np.array(vertices)
        self.vertices = np.unique(self.vertices, axis=0)
        c_hull = ConvexHull(self.vertices)

        # Scipy finds a loooot of identical equations; so simplify
        simplified = np.unique(c_hull.equations, axis=0)
        A = simplified[:, :-1]
        b = -simplified[:, -1]

        self.conv_hull = MyPolytope(A, b)

if __name__ == "__main__":
    from models.sys_model import SystemModel
    from util.broken_thruster import BrokenThruster
    from util.polytope import MyPolytope
    model = SystemModel(dt=0.1)

    model.set_fault(BrokenThruster(0, 1.0))
    model.set_fault(BrokenThruster(1, 1.0))
    model.set_fault(BrokenThruster(4, 1.0))
    model.set_fault(BrokenThruster(5, 1.0))
    # model.set_fault(BrokenThruster(6, 1.0))
    # model.set_fault(BrokenThruster(7, 1.0))
    # model.set_fault(BrokenThruster(8, 0.0))
    # model.set_fault(BrokenThruster(11, 0.0))

    input_bounds = InputBounds(model)
    v = input_bounds.get_vertices()

    pol = MyPolytope.from_vertices([vert[0:3] for vert in v])
    pol.plot(title="Forces")

    pol = MyPolytope.from_vertices([vert[3:6] for vert in v])
    pol.plot(title="Torques")

