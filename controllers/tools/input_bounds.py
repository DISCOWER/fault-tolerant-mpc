import numpy as np

class InputBounds:
    """
    For now: dummy class that gives just "some" input bounds
    """

    def get_conv_hull(self):
        """
        Get the convex hull of the input bounds

        Returns for not box constraints in 6d

        Returns:
            tuple: A, b
        """
        print("WARNING: Using dummy input bounds")
        return self.box_constraints_polytope()
                         
    def box_constraints_polytope(self, lower_bounds=None, upper_bounds=None):
        """
        Create a polytope for the box constraints
        """
        # Set defaults if not provided
        if lower_bounds is None:
            lower_bounds = -np.ones(6)
        else:
            lower_bounds = np.array(lower_bounds, dtype=float)
            
        if upper_bounds is None:
            upper_bounds = np.ones(6)
        else:
            upper_bounds = np.array(upper_bounds, dtype=float)
        
        # Check dimensions
        if len(lower_bounds) != 6 or len(upper_bounds) != 6:
            raise ValueError("Both lower and upper bounds must have exactly 6 elements")
        
        # For each dimension, we need two constraints:
        # x_i >= lower_i (or -x_i <= -lower_i)
        # x_i <= upper_i
        
        # Create identity and negative identity matrices for the constraints
        A_upper = np.eye(6)  # For x_i <= upper_i
        A_lower = -np.eye(6)  # For x_i >= lower_i (or -x_i <= -lower_i)
        
        # Combine the constraint matrices
        A = np.vstack((A_upper, A_lower))
        
        # Create the corresponding b vector
        b_upper = upper_bounds
        b_lower = -lower_bounds  # Negative because we converted x_i >= lower_i to -x_i <= -lower_i
        b = np.concatenate((b_upper, b_lower))
        
        return A, b