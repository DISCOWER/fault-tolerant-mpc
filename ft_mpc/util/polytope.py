import numpy as np
import matplotlib.pyplot as plt
import polytope
from scipy.spatial import ConvexHull, HalfspaceIntersection
import cvxpy as cp
import itertools

class MyPolytope:
    """
    A certainly-not-optimized implementation of a polytope class that combines the capabilities
    of the polytope package, cvxpy and scipy.spatial.
    """
    def __init__(self, A, b):
        # Check inputs
        assert A.shape[0] == b.shape[0]
        self.A = A
        self.b = b
        self.Nx = A.shape[1]
        self.Nc = A.shape[0]

    @classmethod
    def from_box(cls, lower, upper):
        A = np.vstack((np.eye(len(lower)), -np.eye(len(lower))))
        b = np.vstack((np.array(upper).reshape(-1,1), -np.array(lower).reshape(-1,1)))
        return cls(A, b)

    @classmethod
    def from_vertices(cls, vertices):
        hull = ConvexHull(vertices)
        A = hull.equations[:, :-1]
        b = -hull.equations[:, -1]
        return cls(A, b)

    def contains(self, x):
        return np.all( (self.A @ x).reshape(-1,1) <= self.b )

    def largest_contained_box(self):
        """
        Calculate the largest box that is contained in the polytope.
        returns upper bounds, lower bounds
        """
        ubs = cp.Variable(self.A.shape[1], name='ubs')
        lbs = cp.Variable(self.A.shape[1], name='lbs')

        constraints = []
        for i in range(self.A.shape[1]):
            constraints.append(ubs[i] >= lbs[i])

        combinations = list(itertools.product(*[[ubs[i], lbs[i]] for i in range(self.A.shape[1])]))
        for combination in combinations:
            print(combination)
            constraints.append(self.A @ cp.vstack(combination) <= self.b)
        
        obj = 0
        for i in range(self.A.shape[1]):
            obj += cp.log(ubs[i] - lbs[i])

        objective = cp.Minimize(-obj)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)
        variables = problem.solution.primal_vars

        return ubs.value, lbs.value

    def largest_contained_ball(self):
        """
        Calculate the largest ball that is contained in the polytope.
        returns center, radius
        """
        poly = polytope.Polytope(self.A, self.b)
        return poly.chebXc, poly.chebR

    def is_bounded(self):
        """
        Check if the normal vectors of the hyperplanes form a convex hull that contains zero 
        in its interior
        """
        raise NotImplementedError

    def vertices(self):
        """
        Compute the vertices of the polytope
        """
        # Find a feasible point within the polytope
        poly = polytope.Polytope(self.A, self.b)
        feasible_point = poly.chebXc
        conv_hull = np.concatenate((self.A, -self.b.reshape(-1,1)), axis=1)
        # Find the vertices of the polytope
        hs = HalfspaceIntersection(conv_hull, feasible_point)
        vertices = hs.intersections

        return vertices

    def reduce_dims(self):
        """
        Reduce the dimensionality of the polytope by removing redundant constraints.
        Method as described in https://www.cs.mcgill.ca/~fukuda/soft/polyfaq/node24.html
        """
        # Remove zero rows in A (can occur due to projections)
        mask = np.any(self.A != 0, axis=1)
        print(mask)
        self.A = self.A[mask, :]
        if self.b.ndim == 1:
            self.b = self.b[mask]
        else:
            self.b = self.b[mask, :]
        self.Nc = self.A.shape[0]

        # Now remove actual redundant constraints
        i = 0
        while self.Nc > i:
            # Check for redundancy
            active_constraints = [j for j in range(self.Nc) if i != j]
            A_active = self.A[active_constraints, :]
            b = self.b.flatten()
            b_active = b[active_constraints]

            s = self.A[i, :]
            t = b[i]

            x = cp.Variable(self.Nx)

            obj = cp.Maximize(s @ x)

            constr = []
            for j in range(len(active_constraints)):
                constr.append(A_active[j, :] @ x <= b_active[j])
            constr.append(s @ x <= t + 10)

            prob = cp.Problem(obj, constr)
            prob.solve(solver=cp.SCS, verbose=False)

            try:
                if prob.value <= t:
                    # Constraint i is redundant
                    self.A = np.delete(self.A, i, axis=0)
                    self.b = np.delete(self.b, i, axis=0)
                    self.Nc -= 1
                else:
                    # constraint is necessary
                    i += 1
            except:
                print("XXXX FAILED XXXX")
                print(A_active / np.abs(b_active.reshape(-1,1)))
                print(s / t)
                self.plot_lines(A_active, b_active)
                exit()

    def minkowski_subtract_circle(self, r):
        """
        Calculate P ominus {x | ||x|| <= r/2}
        """
        return MyPolytope(self.A, self.b - np.linalg.norm(self.A, axis=1) * r)
        # return MyPolytope(self.A, self.b - np.linalg.norm(self.A, axis=1) * np.sign(self.b) * r)

    def minkowski_add_vector(self, v):
        """
        Calculate P oplus {v}
        """
        verts = self.vertices()
        new_verts = verts + v
        return MyPolytope.from_vertices(new_verts)

    def set_subtraction_along_vector(self, v):
        """
        Subtract parts of the polytope in the direction of the vector v
        """
        return MyPolytope(self.A, self.b - np.abs(np.dot(self.A, v)))
        # return MyPolytope(self.A, self.b - np.abs(np.dot(self.A, v)) * np.sign(self.b))

    def __str__(self):
        rep = ("Polytope with A =\n" + np.array2string(self.A).replace('\n', ';') + " and \n" + 
            "b = \n" + np.array2string(self.b).replace('\n', ';'))
        return rep

    def plot_slice(self, fixed_values, *args, **kwargs):
        """
        Plot a 2D projection of an N-dimensional polytope by fixing N-2 dimensions
        Format of fixed_values: {dim1: val1, dim2: val2, ...}
        """
        fixed_dims = list(fixed_values.keys())
        free_dims = [i for i in range(self.Nx) if i not in fixed_dims]

        new_A = self.A[:, free_dims]
        fixed_vec = np.zeros(self.Nx)
        for i in range(self.Nx):
            if i in fixed_values.keys():
                fixed_vec[i] = fixed_values[i]
        new_b = self.b.flatten() - (self.A @ fixed_vec).flatten()

        print("new A:")
        print(new_A)
        print("new b:")
        print(new_b)

        new_p = MyPolytope(new_A, new_b)
        # self.plot_lines(new_A, new_b, *args, **kwargs)
        new_p.reduce_dims()
        print(new_p)
        new_p.plot(*args, **kwargs)

    def plot_lines(self, A, b, axs=None, color=None, set_aspect_equal=None):
        show = False
        if axs is None:
            fig, axs = plt.subplots(1, 1)
            show=True

        color = color if color is not None else np.rand(3)

        if A.shape[1] != 2:
            print("Can only line-plot 2D polytopes")
            return

        for i in range(self.Nc):
            x = np.linspace(-10, 10, 100)
            if A[i, 1] != 0:
                y = (b[i] - A[i, 0] * x) / A[i, 1]
            else:
                x = np.array( [b[i] / A[i, 0]] * 100 )
                y = np.linspace(-10, 10, 100)

            mask = x>=-10
            x = x[mask]
            y = y[mask]
            mask = x<=10
            x = x[mask]
            y = y[mask]
            mask = y>=-10
            x = x[mask]
            y = y[mask]
            mask = y<=10
            x = x[mask]
            y = y[mask]

            axs.plot(x, y, color=color)

        if show:
            plt.show()

    def plot(self, axs=None, color=None, *args, **kwargs):
        if self.Nx != 2 and self.Nx != 3:
            raise NotImplementedError("Plotting is only implemented for 2D and 3D polytopes")

        show = False
        if axs is None:
            fig, axs = plt.subplots(1, 1)
            show=True
            if self.Nx == 3:
                axs = fig.add_subplot(111, projection='3d')


        if self.Nx == 2:
            axs = self._plot_2d_polygon(axs, color, *args, **kwargs)
        elif self.Nx == 3:
            axs = self._plot_3d_polytope(axs, color, *args, **kwargs)

        if show:
            plt.show()
        else:
            return axs

    def _plot_2d_polygon(self, ax, color=None, *args, **kwargs):
        if self.Nx != 2:
            raise Exception(f"This ploytope is {self.Nx}D. You called a method for 2D.")
        intersections = self.vertices()

        if len(intersections) > 0:
            # Find the centroid of the intersections
            centroid = np.mean(intersections, axis=0)

            # Sort the intersections by their angle from the centroid
            sorted_points = sorted(intersections, key=lambda p: np.arctan2(*(p - centroid)[::-1]))
            sorted_points.append(sorted_points[0])  # Close the polygon

            # Plot the polygon
            points = np.array(sorted_points)
            color = color if color is not None else 'b'
            ax.plot(points[:, 0], points[:, 1], color=color)
            # ax.fill(points[:, 0], points[:, 1], color=color)
            ax.fill(points[:, 0], points[:, 1], color=color, alpha=0.3)

        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        else:
            ax.set_title('2D Polygon: Ax <= b')

        # Set equal aspect ratio if not explicitly disabled
        if not('set_aspect_equal' in kwargs and not(kwargs['set_aspect_equal'])):
            ax.set_aspect('equal')

        # Add grid
        ax.grid(True)

        return ax

    def _plot_3d_polytope(self, ax, color=None, *args, **kwargs):
        vertices = self.vertices()
        conv_hull = ConvexHull(vertices)

        show = ax is None
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Plot defining corner points
        ax.plot(vertices.T[0], vertices.T[1], vertices.T[2], "ko")

        # Plot edges
        for s in conv_hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            if color is None:
                ax.plot(vertices[s, 0], vertices[s, 1], vertices[s, 2])
            else:
                ax.plot(vertices[s, 0], vertices[s, 1], vertices[s, 2], color)
        
        # Plot planes (make the plot easier to interpret)
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        xy_plane_x, xy_plane_y = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(y), max(y), 10))
        yz_plane_y, yz_plane_z = np.meshgrid(np.linspace(min(y), max(y), 10), np.linspace(min(z), max(z), 10))
        zx_plane_x, zx_plane_z = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(z), max(z), 10))

        alpha_val = 0.1
        ax.plot_surface(xy_plane_x, xy_plane_y, np.zeros_like(xy_plane_x), alpha=alpha_val)
        ax.plot_surface(np.zeros_like(yz_plane_y), yz_plane_y, yz_plane_z, alpha=alpha_val)
        ax.plot_surface(zx_plane_x, np.zeros_like(zx_plane_x), zx_plane_z, alpha=alpha_val)

        ax.plot([min(x), max(x)], [0,0], [0,0], "k", alpha=0.8)
        ax.plot([0,0], [min(y), max(y)], [0,0], "k", alpha=0.8)
        ax.plot([0,0], [0,0], [min(z), max(z)], "k", alpha=0.8)

        # axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        else:
            ax.set_title('3D Polytope: Ax <= b')


if __name__=="__main__":
    # Example
    A = np.array([[1, 1], 
                  [-1, 1], 
                  [-1, -1], 
                  [-1, -1], 
                  [1, -1]])
    b = np.array([[1], 
                  [1], 
                  [1], 
                  [0], 
                  [1]])
    p = MyPolytope(A, b)

    print(p.A)
    print(p.b)

    p.reduce_dims()
    print(p.A)
    print(p.b)

    A = np.array([[1, 1], 
                  [-1, 1], 
                  [-1, -1], 
                  [1, -1]])
    b = np.array([[1], 
                  [1], 
                  [1], 
                  [1]])
    p2 = MyPolytope(A, b)

    fig, ax = plt.subplots(1, 1)
    p.plot(ax)
    p2.plot(ax, color='r')
    plt.show()
