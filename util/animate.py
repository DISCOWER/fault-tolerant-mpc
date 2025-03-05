import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def animate_trajectory(positions, quaternions, box_dimensions=(0.5, 1.0, 0.5), 
                       total_time=10.0, save_animation=False, save_path=None):
    """
    Animate a 3D trajectory with a box moving along it, using quaternions for orientation.
    
    Parameters:
    -----------
    positions : array-like
        Array of shape (n, 3) containing the x, y, z positions along the trajectory.
    quaternions : array-like
        Array of shape (n, 4) containing the orientation quaternions [x, y, z, w].
    box_dimensions : tuple, optional
        The dimensions of the box (length, width, height), defaults to (0.5, 1.0, 0.5).
    total_time : float, optional
        The total time of the animation in seconds, defaults to 10.0.
    save_animation : bool, optional
        Whether to save the animation as a file, defaults to False.
    save_path : str, optional
        Path to save the animation file. Required if save_animation is True.
    
    Returns:
    --------
    anim : FuncAnimation
        The animation object.
    """
    # Convert inputs to numpy arrays if they aren't already
    positions = np.array(positions)
    # Calculates internally with [w, x, y, z] quaternions instead of [x, y, z, w]
    quaternions = np.array(quaternions)
    quaternions = np.roll(quaternions, 1, axis=0)
    num_frames = len(positions)
    
    # Create time stamps
    t = np.linspace(0, total_time, num_frames)
    
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
        
        return box
    
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
    ax.set_title('3D Box Moving Along Trajectory')
    
    # Initialize animation
    def init():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        time_text.set_text('')
        
        nonlocal current_cube
        if current_cube is not None:
            current_cube.remove()
            current_cube = None
        
        return [trajectory_line, time_text]
    
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
        current_cube = create_box(pos, box_dimensions, quat)
        ax.add_collection3d(current_cube)
        
        # Update time
        time_text.set_text(f'Time: {t[frame]:.2f}s')
        
        return [trajectory_line, time_text, current_cube]
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                        interval=int(total_time*1000/num_frames), blit=False)
    
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
    
    return positions, np.array(quaternions)

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
    positions, quaternions = generate_sample_trajectory()
    
    # Set box dimensions
    box_dimensions = (0.5, 1.0, 0.5)
    
    # Animate the trajectory
    anim = animate_trajectory(
        positions=positions,
        quaternions=quaternions,
        box_dimensions=box_dimensions,
        total_time=10.0,
        save_animation=False,
        save_path=None
    )
    
    plt.show()