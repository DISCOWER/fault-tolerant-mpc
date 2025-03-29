import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def animate_trajectory(history, time, model=None,
                       box_dimensions=(0.15, 0.15, 0.15), 
                       save_animation=False, save_path=None):
    """
    Animate a 3D trajectory with a box moving along it, using quaternions for orientation.
    
    Parameters:
    -----------
    positions : array-like
        Array of shape (n, 3) containing the x, y, z positions along the trajectory.
    quaternions : array-like
        Array of shape (n, 4) containing the orientation quaternions [x, y, z, w].
    time: array-like
        Array of shape (n,) containing the time points for each position.
    inputs : array-like, optional
        Array of shape (n, 16) containing the 16-dimensional input for each thruster at each time point.
        If None, no thruster arrows will be displayed.
    faulty_inputs : array-like, optional
        faulty input
    box_dimensions : tuple, optional
        The dimensions of the box (length, width, height), defaults to (0.5, 1.0, 0.5).
    save_animation : bool, optional
        Whether to save the animation as a file, defaults to False.
    save_path : str, optional
        Path to save the animation file. Required if save_animation is True.
    
    Returns:
    --------
    anim : FuncAnimation
        The animation object.
    """
    positions = [h.position for h in history]
    quaternions = [h.orientation for h in history]
    inputs = [h.input for h in history]
    faulty_inputs = [h.faulty_force for h in history]
    desired_trajectory = [h.desired_position for h in history]

    # Convert inputs to numpy arrays if they aren't already
    positions = np.array(positions)
    # Calculates internally with [w, x, y, z] quaternions instead of [x, y, z, w]
    quaternions = np.array(quaternions)
    quaternions = np.roll(quaternions, 1, axis=1)
    num_frames = len(positions)
    
    # Convert desired trajectory to numpy array
    if desired_trajectory is not None:
        desired_trajectory = np.array(desired_trajectory)
    
    # Check if inputs are provided
    if inputs is None:
        inputs = np.zeros((num_frames, 16))
    else:
        inputs = np.array(inputs)
        if inputs.shape[0] != num_frames or inputs.shape[1] != 16:
            raise ValueError(f"inputs should have shape ({num_frames}, 16), got {inputs.shape}")

    if faulty_inputs is not None:
        faulty_inputs = np.array(faulty_inputs)

    # Define thruster positions and directions
    d1 = 0.15
    d2 = 0.12
    d3 = 0.09
    d4 = 0.05

    thruster_positions = {
         0: ( d1,  d2,  d4),
         1: ( d1,  d2, -d4),
         2: (-d1,  d2,  d4),
         3: (-d1,  d2, -d4),
         4: ( d1, -d2,  d4),
         5: ( d1, -d2, -d4),
         6: (-d1, -d2,  d4),
         7: (-d1, -d2, -d4),
         8: ( d3,  d1,  0),
         9: (-d3,  d1,  0),
        10: ( d3, -d1,  0),
        11: (-d3, -d1,  0),
        12: (  0,  d2,  d1),
        13: (  0,  d2, -d1),
        14: (  0, -d2,  d1),
        15: (  0, -d2, -d1)
    }

    thruster_directions = {
         0: ( 1,  0,  0),
         1: ( 1,  0,  0),
         2: (-1,  0,  0),
         3: (-1,  0,  0),
         4: ( 1,  0,  0),
         5: ( 1,  0,  0),
         6: (-1,  0,  0),
         7: (-1,  0,  0),
         8: ( 0,  1,  0),
         9: ( 0,  1,  0),
        10: ( 0, -1,  0),
        11: ( 0, -1,  0),
        12: ( 0,  0,  1),
        13: ( 0,  0, -1),
        14: ( 0,  0,  1),
        15: ( 0,  0, -1)
    }

    for i in range(16):
        print(np.cross(thruster_positions[i], thruster_directions[i]))
    
    # Create time stamps
    t = time
    total_time = t[-1]
    
    # Create the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract position data
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    # Initialize plot elements
    trajectory_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.5, label='Trajectory')
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    
    # Variable to store the current cube visualization
    current_cube = None
    
    # Variable to store current thruster arrows
    thruster_arrows = []
    
    # Add elements for desired setpoint and center point trajectory
    desired_setpoint, = ax.plot([], [], [], 'ro', markersize=6, label='Desired Setpoint')
    center_point_trajectory, = ax.plot([], [], [], 'k--', linewidth=2, alpha=0.7, label='Center Point Trajectory')
    
    # Function to create a box with proper orientation using quaternions
    def create_box(center, dimensions, quaternion):
        # Unpack box dimensions
        e1, e2, e3 = dimensions
        
        # Define the 8 vertices of a box centered at origin
        vertices = np.array([
            [-e1, -e2, -e3], [e1, -e2, -e3], [e1, e2, -e3], [-e1, e2, -e3],
            [-e1, -e2, e3], [e1, -e2, e3], [e1, e2, e3], [-e1, e2, e3]
        ])
        
        # Convert quaternion to rotation matrix
        qw, qx, qy, qz = quaternion
        rotation_matrix = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
        ])
        
        # Apply the orientation (rotation)
        rotated_vertices = np.dot(vertices, rotation_matrix.T)
        
        # Translate to center position
        transformed_vertices = rotated_vertices + center
        
        # Define the 6 faces of the box
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
        ]
        
        # Get the coordinates for each face
        face_vertices = [[transformed_vertices[face_idx] for face_idx in face] for face in faces]
        
        # Colors for each face to make orientation more visible
        colors = ['#2978A0', '#1F6E8C', '#0A536F', '#1D3E53', '#355070', '#293241']
        
        # Create a Poly3DCollection with face colors
        box = Poly3DCollection(face_vertices, alpha=0.7, linewidths=1, edgecolor='black', 
                              facecolors=colors)
        
        return box, rotation_matrix, transformed_vertices
    
    # Find the bounding box of the trajectory to set appropriate axis limits
    x_min, x_max = np.min(x) - 1, np.max(x) + 1
    y_min, y_max = np.min(y) - 1, np.max(y) + 1
    z_min, z_max = np.min(z) - 1, np.max(z) + 1
    
    # Setup the axes
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Spacecraft Moving Along Trajectory with Thrusters')
    
    # Create arrow artists for each thruster
    for i in range(16):
        arrow, = ax.plot([], [], [], 'r-', linewidth=2, alpha=0.0)
        thruster_arrows.append(arrow)

    # Create arrows for local coordinate system
    local_x_arrow, = ax.plot([], [], [], 'r-', linewidth=3)
    local_y_arrow, = ax.plot([], [], [], 'g-', linewidth=3)
    local_z_arrow, = ax.plot([], [], [], 'b-', linewidth=3) 

    # Create artists for center point
    center_point, = ax.plot([], [], [], 'k--', linewidth=2)
    
    # Store center point positions for trajectory
    center_point_positions = []

    # Initialize animation
    def init():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        time_text.set_text('')
        
        # Initialize desired setpoint
        desired_setpoint.set_data([], [])
        desired_setpoint.set_3d_properties([])
        
        # Initialize center point trajectory
        center_point_trajectory.set_data([], [])
        center_point_trajectory.set_3d_properties([])
        
        nonlocal current_cube
        if current_cube is not None:
            current_cube.remove()
            current_cube = None
        
        # Initialize all thruster arrows
        for arrow in thruster_arrows:
            arrow.set_data([], [])
            arrow.set_3d_properties([])

        # Initialize coordinate system arrows
        local_x_arrow.set_data([], [])
        local_x_arrow.set_3d_properties([])
        local_y_arrow.set_data([], [])
        local_y_arrow.set_3d_properties([])
        local_z_arrow.set_data([], [])
        local_z_arrow.set_3d_properties([])

        # Initialize center point
        center_point.set_data([], [])
        center_point.set_3d_properties([])
        
        return [trajectory_line, time_text, desired_setpoint, center_point_trajectory, 
                local_x_arrow, local_y_arrow, local_z_arrow, center_point] + thruster_arrows
    
    # Animation update function
    def update(frame):
        # Show the trajectory
        trajectory_line.set_data(x[:frame+1], y[:frame+1])
        trajectory_line.set_3d_properties(z[:frame+1])
        
        # Current position and orientation
        pos = positions[frame]
        quat = quaternions[frame]
        
        # Clear previous box
        nonlocal current_cube
        if current_cube is not None:
            current_cube.remove()
        
        # Create new box at current position with proper orientation
        current_cube, rotation_matrix, transformed_vertices = create_box(pos, box_dimensions, quat)
        ax.add_collection3d(current_cube)
        
        # Update thruster arrows
        for i in range(16):
            # Get thruster position and direction in local coordinates
            local_pos = np.array(thruster_positions[i])
            local_dir = np.array(thruster_directions[i])
            
            # Apply rotation to thruster position and direction
            rotated_pos = np.dot(local_pos, rotation_matrix.T)
            rotated_dir = np.dot(local_dir, rotation_matrix.T)
            
            # Get global position
            global_pos = rotated_pos + pos
            
            # Get input value for this thruster
            input_val = inputs[frame, i]
            is_faulty = False

            if faulty_inputs is not None:
                if inputs[frame, i] < 1e-5 and faulty_inputs[frame, i] >= 1e-5:
                    input_val = faulty_inputs[frame, i]
                    is_faulty = True
            
            # Set arrow properties based on input
            if abs(input_val) > 1e-5:
                # Scale arrow length based on input
                arrow_length = min(max(0.1, abs(input_val)), 1.0)  # Limit between 0.1 and 1.0
                
                # Calculate end point of arrow
                end_point = global_pos + rotated_dir * arrow_length
                
                # Set arrow data
                thruster_arrows[i].set_data([global_pos[0], end_point[0]], [global_pos[1], end_point[1]])
                thruster_arrows[i].set_3d_properties([global_pos[2], end_point[2]])
                
                # Set arrow opacity
                thruster_arrows[i].set_alpha(1.0)
                
                # Color based on thruster input intensity
                color = "red" if is_faulty else "green"
                thruster_arrows[i].set_color(color)
            else:
                # Make arrow invisible for small inputs
                thruster_arrows[i].set_alpha(0.0)

        # Update local coordinate system arrows
        arrow_length = 0.75  # Length of coordinate arrows
        
        # Update x-axis arrow (red)
        x_dir = np.array([1, 0, 0])
        rotated_x = np.dot(x_dir, rotation_matrix.T)
        local_x_arrow.set_data([pos[0], pos[0] + rotated_x[0] * arrow_length], 
                              [pos[1], pos[1] + rotated_x[1] * arrow_length])
        local_x_arrow.set_3d_properties([pos[2], pos[2] + rotated_x[2] * arrow_length])
        
        # Update y-axis arrow (green)
        y_dir = np.array([0, 1, 0])
        rotated_y = np.dot(y_dir, rotation_matrix.T)
        local_y_arrow.set_data([pos[0], pos[0] + rotated_y[0] * arrow_length], 
                              [pos[1], pos[1] + rotated_y[1] * arrow_length])
        local_y_arrow.set_3d_properties([pos[2], pos[2] + rotated_y[2] * arrow_length])
        
        # Update z-axis arrow (blue)
        z_dir = np.array([0, 0, 1])
        rotated_z = np.dot(z_dir, rotation_matrix.T)
        local_z_arrow.set_data([pos[0], pos[0] + rotated_z[0] * arrow_length], 
                              [pos[1], pos[1] + rotated_z[1] * arrow_length])
        local_z_arrow.set_3d_properties([pos[2], pos[2] + rotated_z[2] * arrow_length]) 

        local_x_arrow.set_alpha(0.5)
        local_y_arrow.set_alpha(0.5)
        local_z_arrow.set_alpha(0.5)

        # Update center point and its trajectory
        center_pt_position = None
        if model is not None:
            rotated_c = np.dot(model.spiral_params.r, rotation_matrix.T)
            center_pt_position = pos + rotated_c
            center_point.set_data([pos[0], center_pt_position[0]],
                                  [pos[1], center_pt_position[1]])
            center_point.set_3d_properties([pos[2], center_pt_position[2]])
            center_point.set_alpha(1.0)
            
            # Add current center point position to trajectory
            if frame > 0:  # Only add if we're past the first frame
                # Only add the position if we haven't seen this frame before
                if len(center_point_positions) <= frame:
                    center_point_positions.append(center_pt_position)
                
                # Update center point trajectory - only show up to current frame
                if center_point_positions:
                    cp_trajectory = np.array(center_point_positions)
                    center_point_trajectory.set_data(cp_trajectory[:, 0], cp_trajectory[:, 1])
                    center_point_trajectory.set_3d_properties(cp_trajectory[:, 2])
        else:
            center_point.set_alpha(0.0)
            
        # Update desired setpoint visualization
        if desired_trajectory is not None:
            desired_pos = desired_trajectory[frame]
            desired_setpoint.set_data([desired_pos[0]], [desired_pos[1]])
            desired_setpoint.set_3d_properties([desired_pos[2]])
            desired_setpoint.set_alpha(1.0)

        # Update time
        time_text.set_text(f'Time: {t[frame]:.2f}s')
        
        return [trajectory_line, time_text, current_cube, center_point, center_point_trajectory, 
                desired_setpoint] + thruster_arrows
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                        interval=int(total_time*1000/num_frames), blit=False)
    
    # Add a legend
    handles = [trajectory_line, center_point_trajectory, desired_setpoint]
    labels = ['Trajectory', 'Center Point Trajectory', 'Desired Setpoint']
    ax.legend(handles=handles, labels=labels, loc='upper right')
    
    # Save the animation if requested
    if save_animation:
        if save_path is None:
            raise ValueError("save_path must be provided if save_animation is True")
        
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=30)
        elif save_path.endswith('.mp4'):
            from matplotlib import animation
            anim.save(save_path, writer=animation.FFMpegWriter(fps=30))
        else:
            raise ValueError("save_path must end with '.gif' or '.mp4'")
    
    plt.tight_layout()

    plt.show()
    return anim

def generate_sample_trajectory():
    """Generate a sample helical trajectory with quaternion orientations."""
    # Number of sample points
    num_points = 200
    
    # Generate time points
    t = np.linspace(0, 10, num_points)
    
    # Helix parameters
    radius = 1.0
    height = 5.0
    
    # Positions along a helix
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = height * t / 10
    
    positions = np.column_stack((x, y, z))
    
    # Generate quaternions that make the box point in the direction of motion and roll
    quaternions = []
    
    # Calculate direction vectors for each point
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    dz = np.gradient(z, t)
    
    # Normalize direction vectors
    directions = np.column_stack((dx, dy, dz))
    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    
    for i, direction in enumerate(directions):
        # We want the box to point in the direction of motion
        forward = direction
        
        # Construct a frame with z axis aligned with the forward direction
        # First, find a perpendicular vector (any will do)
        if abs(forward[0]) < abs(forward[1]):
            right = np.array([0, forward[2], -forward[1]])
        else:
            right = np.array([forward[2], 0, -forward[0]])
        right = right / np.linalg.norm(right)
        
        # Complete the frame with the third axis
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        
        # Add some rolling around the forward axis
        roll_angle = t[i] * 2  # Rolling speed
        cos_roll = np.cos(roll_angle)
        sin_roll = np.sin(roll_angle)
        
        right_rolled = right * cos_roll + up * sin_roll
        up_rolled = -right * sin_roll + up * cos_roll
        
        # Construct the rotation matrix with axes as columns
        rotation_matrix = np.column_stack((right_rolled, up_rolled, forward))
        
        # Convert rotation matrix to quaternion
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
        quaternions.append(quaternion)
    
    return positions, np.array(quaternions), t

def generate_sample_inputs(num_points):
    """Generate sample inputs for thrusters."""
    inputs = np.zeros((num_points, 16))
    
    # Generate some sample thruster patterns
    for i in range(num_points):
        # Make some thrusters active at different times
        t_norm = i / num_points
        
        # X-axis thrusters (0-7)
        if t_norm < 0.25:
            inputs[i, 0:2] = 0.5 * np.sin(t_norm * 8 * np.pi)
        elif t_norm < 0.5:
            inputs[i, 2:4] = 0.5 * np.sin((t_norm - 0.25) * 8 * np.pi)
        
        # Y-axis thrusters (8-11)
        inputs[i, 8:12] = 0.3 * np.sin(t_norm * 4 * np.pi)
        
        # Z-axis thrusters (12-15)
        if t_norm > 0.5:
            pattern = np.sin(t_norm * 12 * np.pi)
            inputs[i, 12:16] = pattern * np.array([0.7, 0.5, 0.7, 0.5])
    
    return inputs

def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    
    return np.array([qw, qx, qy, qz])

if __name__ == '__main__':
    # Generate sample trajectory and quaternions
    positions, quaternions, time = generate_sample_trajectory()
    
    # Generate sample thruster inputs
    inputs = generate_sample_inputs(len(positions))
    
    # Set box dimensions
    box_dimensions = (0.5, 1.0, 0.5)
    
    # Animate the trajectory
    anim = animate_trajectory(
        positions=positions,
        quaternions=quaternions,
        time=time,
        inputs=inputs,
        box_dimensions=box_dimensions,
        save_animation=False,
        save_path=None
    )
    
    plt.show()
