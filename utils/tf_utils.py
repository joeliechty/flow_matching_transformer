import torch

###########################
# --- PRIVATE METHODS --- #
###########################
# rotation conversions
def _quat_to_rot_mat(quaternions, w_first=True):
    """
    Highly optimized batch conversion of quaternions to rotation matrices.
    Uses vectorized operations for maximum performance on large batches.
    
    Args:
        quaternions (torch.Tensor): A tensor of shape (N, 4) or (..., 4) representing quaternions (qw, qx, qy, qz).
        
    Returns:
        torch.Tensor: A tensor of shape (N, 3, 3) or (..., 3, 3) representing rotation matrices.
    """
    # Ensure input is at least 2D for batch processing
    original_shape = quaternions.shape
    if quaternions.dim() == 1:
        quaternions = quaternions.unsqueeze(0)
        was_1d = True
    else:
        was_1d = False
    
    # Reshape to (batch_size, 4) for efficient processing
    batch_size = quaternions.shape[0]
    quaternions = quaternions.view(-1, 4)
    
    # Normalize quaternions for numerical stability
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
    
    if not w_first:
        # If w is not first, rearrange to (qw, qx, qy, qz)
        quaternions = quaternions[:, 1:4].contiguous()

    # Extract components
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Precompute all required terms using broadcasting
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    # Build rotation matrices using vectorized operations
    batch_rot_matrices = torch.stack([
        torch.stack([w2 + x2 - y2 - z2, 2*(xy - wz), 2*(xz + wy)], dim=1),
        torch.stack([2*(xy + wz), w2 - x2 + y2 - z2, 2*(yz - wx)], dim=1),
        torch.stack([2*(xz - wy), 2*(yz + wx), w2 - x2 - y2 + z2], dim=1)
    ], dim=1)
    
    # Reshape back to original batch dimensions
    if was_1d:
        return batch_rot_matrices.squeeze(0)
    else:
        target_shape = original_shape[:-1] + (3, 3)
        return batch_rot_matrices.view(target_shape)

def _rot_mat_to_quat(rotation_matrices, w_first=True):
    """
    Convert a batch of rotation matrices to quaternions using Shepperd's method for numerical stability.
    
    Args:
        rotation_matrices (torch.Tensor): A tensor of shape (..., 3, 3) representing rotation matrices.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 4) representing quaternions (qw, qx, qy, qz).
    """
    batch_shape = rotation_matrices.shape[:-2]
    device = rotation_matrices.device
    dtype = rotation_matrices.dtype
    
    # Flatten batch dimensions for easier processing
    R = rotation_matrices.view(-1, 3, 3)
    batch_size = R.shape[0]
    
    # Initialize output quaternions
    q = torch.zeros((batch_size, 4), device=device, dtype=dtype)
    
    # Extract diagonal elements
    R00, R11, R22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    
    # Compute trace
    trace = R00 + R11 + R22
    
    # Use Shepperd's method for numerical stability
    # Case 1: trace > 0 (most common case)
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
        q[mask1, 0] = 0.25 * s  # qw
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # qx
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # qy
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # qz
    
    # Case 2: R00 > R11 and R00 > R22
    mask2 = (~mask1) & (R00 > R11) & (R00 > R22)
    if mask2.any():
        s = torch.sqrt(1.0 + R00[mask2] - R11[mask2] - R22[mask2]) * 2  # s = 4 * qx
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s  # qw
        q[mask2, 1] = 0.25 * s  # qx
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s  # qy
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s  # qz
    
    # Case 3: R11 > R22
    mask3 = (~mask1) & (~mask2) & (R11 > R22)
    if mask3.any():
        s = torch.sqrt(1.0 + R11[mask3] - R00[mask3] - R22[mask3]) * 2  # s = 4 * qy
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s  # qw
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s  # qx
        q[mask3, 2] = 0.25 * s  # qy
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s  # qz
    
    # Case 4: else (R22 is largest)
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R22[mask4] - R00[mask4] - R11[mask4]) * 2  # s = 4 * qz
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s  # qw
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s  # qx
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s  # qy
        q[mask4, 3] = 0.25 * s  # qz
    
    # Ensure quaternion is normalized and positive w (canonical form)
    q = q / torch.norm(q, dim=-1, keepdim=True)
    
    # Make quaternion canonical (positive w)
    negative_w = q[:, 0] < 0
    q[negative_w] = -q[negative_w]
    
    # Reshape back to original batch shape
    quat = q.view(*batch_shape, 4)

    if not w_first:
        # If w was not first, rearrange to (qx, qy, qz, qw)
        quat = quat[..., 1:4].contiguous()
    
    return quat

def _ortho6d_to_quat(ortho6d, w_first=True):
    """
    Convert a batch of 6-DOF rotation representations (x and y vectors) back to quaternions.
    
    Args:
        ortho6d (torch.Tensor): A tensor of shape (..., 6) representing 6-DOF rotations (x1, x2, x3, y1, y2, y3).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 4) representing quaternions (qw, qx, qy, qz).
    """
    # Extract x and y vectors
    x_vectors = ortho6d[..., :3]  # First three elements
    y_vectors = ortho6d[..., 3:]   # Last three elements

    # Compute the z vector using the cross product
    z_vectors = torch.cross(x_vectors, y_vectors, dim=-1)

    # Recompute the y vector using the cross product to ensure orthogonality
    y_vectors = torch.cross(z_vectors, x_vectors, dim=-1)
    
    # Normalize the vectors
    x_norm = torch.norm(x_vectors, dim=-1, keepdim=True)
    y_norm = torch.norm(y_vectors, dim=-1, keepdim=True)
    z_norm = torch.norm(z_vectors, dim=-1, keepdim=True)

    # Handle potential division by zero
    x_norm = torch.clamp(x_norm, min=1e-8)
    y_norm = torch.clamp(y_norm, min=1e-8)
    z_norm = torch.clamp(z_norm, min=1e-8)

    x_unit = x_vectors / x_norm
    y_unit = y_vectors / y_norm
    z_unit = z_vectors / z_norm

    # Construct rotation matrix from orthonormal vectors
    batch_shape = ortho6d.shape[:-1]
    rotation_matrices = torch.zeros((*batch_shape, 3, 3), dtype=ortho6d.dtype, device=ortho6d.device)
    
    rotation_matrices[..., 0, :] = x_unit  # First column
    rotation_matrices[..., 1, :] = y_unit  # Second column
    rotation_matrices[..., 2, :] = z_unit  # Third column

    # Convert rotation matrices to quaternions using the batch function
    return _rot_mat_to_quat(rotation_matrices, w_first=w_first)

def _quat_to_ortho6d(quaternions, w_first=True):
    """
    Convert a batch of quaternions to 6-DOF rotation representations (x and y vectors of the rotation).
    
    Args:
        quaternions (torch.Tensor): A tensor of shape (..., 4) representing quaternions (qw, qx, qy, qz).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 6) representing the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
    """
    # Get the rotation matrices for the batch of quaternions
    rotation_matrices = _quat_to_rot_mat(quaternions, w_first=w_first)

    # Extract the x and y vectors from the rotation matrices
    x_vector = rotation_matrices[..., 0, :]  # First column
    y_vector = rotation_matrices[..., 1, :]  # Second column

    # Concatenate the x and y vectors to form the 6-DOF representation
    ortho6d = torch.cat([x_vector, y_vector], dim=-1)   

    return ortho6d

def _rot_mat_to_ortho6d(rotation_matrices, w_first=True):
    """
    Convert a batch of rotation matrices to 6-DOF rotation representations (x and y vectors of the rotation).
    
    Args:
        rotation_matrices (torch.Tensor): A tensor of shape (..., 3, 3) representing rotation matrices.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 6) representing the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
    """
    # Extract the x and y vectors from the rotation matrices
    x_vector = rotation_matrices[..., 0, :]  # First column
    y_vector = rotation_matrices[..., 1, :]  # Second column

    # Concatenate the x and y vectors to form the 6-DOF representation
    ortho6d = torch.cat([x_vector, y_vector], dim=-1)   

    return ortho6d

def _ortho6d_to_rot_mat(ortho6d, w_first=True):
    """
    Convert a batch of 6-DOF rotation representations (x and y vectors) to rotation matrices.
    
    Args:
        ortho6d (torch.Tensor): A tensor of shape (..., 6) representing 6-DOF rotations (x1, x2, x3, y1, y2, y3).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing rotation matrices.
    """
    # Extract x and y vectors
    x_vectors = ortho6d[..., :3]  # First three elements
    y_vectors = ortho6d[..., 3:]   # Last three elements

    # Compute the z vector using the cross product
    z_vectors = torch.cross(x_vectors, y_vectors, dim=-1)

    # Recompute the y vector using the cross product to ensure orthogonality
    y_vectors = torch.cross(z_vectors, x_vectors, dim=-1)
    
    # Normalize the vectors
    x_norm = torch.norm(x_vectors, dim=-1, keepdim=True)
    y_norm = torch.norm(y_vectors, dim=-1, keepdim=True)
    z_norm = torch.norm(z_vectors, dim=-1, keepdim=True)

    # Handle potential division by zero
    x_norm = torch.clamp(x_norm, min=1e-8)
    y_norm = torch.clamp(y_norm, min=1e-8)
    z_norm = torch.clamp(z_norm, min=1e-8)

    x_unit = x_vectors / x_norm
    y_unit = y_vectors / y_norm
    z_unit = z_vectors / z_norm

    # Construct rotation matrices from orthonormal vectors
    batch_shape = ortho6d.shape[:-1]
    rotation_matrices = torch.zeros((*batch_shape, 3, 3), dtype=ortho6d.dtype, device=ortho6d.device)
    
    rotation_matrices[..., 0, :] = x_unit  # First column
    rotation_matrices[..., 1, :] = y_unit  # Second column
    rotation_matrices[..., 2, :] = z_unit  # Third column

    return rotation_matrices

# transformation conversions
def _pose_quat_to_T(pose_quat, w_first=True):
    """
    Convert a batch of position and quaternion tensors to transformation matrices.
    Args:
        pose_quat (torch.Tensor): A tensor of shape (..., 7) where the first 3 elements are position (x, y, z)
                                     and the last 4 elements are the quaternion (qw, qx, qy, qz).
    Returns:
        torch.Tensor: A tensor of shape (..., 4, 4) representing transformation matrices.
    """
    position = pose_quat[..., :3]  # First three elements are position
    quaternion = pose_quat[..., 3:]  # Last four elements are quaternion

    # Convert quaternion to rotation matrix
    rotation_matrix = _quat_to_rot_mat(quaternion, w_first=w_first)

    # Create transformation matrices
    batch_shape = position.shape[:-1]
    T = torch.zeros((*batch_shape, 4, 4), dtype=position.dtype, device=position.device)
    
    T[..., :3, :3] = rotation_matrix
    T[..., :3, 3] = position
    T[..., 3, 3] = 1.0  # Homogeneous coordinate
    
    return T

def _T_to_pose_quat(T, w_first=True):
    """
    Convert a batch of transformation matrices to position and quaternion tensors.
    
    Args:
        T (torch.Tensor): A tensor of shape (..., 4, 4) representing transformation matrices.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 7) where the first 3 elements are position (x, y, z)
                      and the last 4 elements are the quaternion (qw, qx, qy, qz).
    """
    position = T[..., :3, 3]  # Extract position (x, y, z)
    
    # Extract rotation matrix from transformation matrix
    rotation_matrix = T[..., :3, :3]
    
    # Convert rotation matrix to quaternion
    quaternion = _rot_mat_to_quat(rotation_matrix, w_first=w_first)
    
    # Concatenate position and quaternion
    pose_quat = torch.cat([position, quaternion], dim=-1)
    
    return pose_quat

def _T_to_pose_ortho6d(T, w_first=True):
    """
    Convert a batch of transformation matrices to position and 6-DOF rotation representation.
    
    Args:
        T (torch.Tensor): A tensor of shape (..., 4, 4) representing transformation matrices.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 9) where the first 3 elements are position (x, y, z)
                      and the last 6 elements are the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
    """
    position = T[..., :3, 3]  # Extract position (x, y, z)
    
    # Extract rotation matrix from transformation matrix
    rotation_matrix = T[..., :3, :3]
    
    # Convert rotation matrix to 6-DOF rotation representation
    ortho6d = _rot_mat_to_ortho6d(rotation_matrix, w_first=w_first)
    
    # Concatenate position and 6-DOF rotation
    pose_ortho6d = torch.cat([position, ortho6d], dim=-1)
    
    return pose_ortho6d

def _pose_ortho6d_to_T(pose_ortho6d, w_first=True):
    """
    Convert a batch of position and 6-DOF rotation tensors to transformation matrices.
    
    Args:
        pose_ortho6d (torch.Tensor): A tensor of shape (..., 9) where the first 3 elements are position (x, y, z)
                                          and the last 6 elements are the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 4, 4) representing transformation matrices.
    """
    position = pose_ortho6d[..., :3]  # First three elements are position
    ortho6d = pose_ortho6d[..., 3:]  # Last six elements are 6-DOF rotation

    # Convert 6-DOF rotation representation to rotation matrix
    rotation_matrix = _ortho6d_to_rot_mat(ortho6d, w_first=w_first)

    # Create transformation matrices
    batch_shape = position.shape[:-1]
    T = torch.zeros((*batch_shape, 4, 4), dtype=position.dtype, device=position.device)
    
    T[..., :3, :3] = rotation_matrix
    T[..., :3, 3] = position
    T[..., 3, 3] = 1.0  # Homogeneous coordinate
    
    return T

def _pose_quat_to_pose_ortho6d(pose_quat, w_first=True):
    """
    Convert a batch of position and quaternion tensors to a 6-DOF rotation representation.
    
    Args:
        pose_quat (torch.Tensor): A tensor of shape (..., 7) where the first 3 elements are position (x, y, z)
                                     and the last 4 elements are the quaternion (qw, qx, qy, qz).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 9) where the first 3 elements are position (x, y, z)
                      and the last 6 elements are the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
    """
    position = pose_quat[..., :3]  # First three elements are position
    quaternion = pose_quat[..., 3:]  # Last four elements are quaternion

    # Convert quaternion to 6-DOF rotation representation
    ortho6d = _quat_to_ortho6d(quaternion, w_first=w_first)

    # Concatenate position and 6-DOF rotation
    pose_ortho6d = torch.cat([position, ortho6d], dim=-1)

    return pose_ortho6d

def _pose_ortho6d_to_pose_quat(pose_ortho6d, w_first=True):
    """
    Convert a batch of position and 6-DOF rotation tensors back to position and quaternion representation.
    
    Args:
        pose_ortho6d (torch.Tensor): A tensor of shape (..., 9) where the first 3 elements are position (x, y, z)
                                          and the last 6 elements are the 6-DOF rotation (x1, x2, x3, y1, y2, y3).
        
    Returns:
        torch.Tensor: A tensor of shape (..., 7) where the first 3 elements are position (x, y, z)
                      and the last 4 elements are the quaternion (qw, qx, qy, qz).
    """
    position = pose_ortho6d[..., :3]  # First three elements are position
    ortho6d = pose_ortho6d[..., 3:]  # Last six elements are 6-DOF rotation

    # Convert 6-DOF rotation representation to quaternion
    quaternion = _ortho6d_to_quat(ortho6d, w_first=w_first)

    # Concatenate position and quaternion
    pose_quat = torch.cat([position, quaternion], dim=-1)

    return pose_quat

def _convert_pose_any_to_pose_quat(pose, w_first=True):
    """
    Convert a batch of poses from any supported representation to position and quaternion representation.
    
    Args:
        pose (torch.Tensor): A tensor of shape (..., D) where D is 7 for 'quat' and 9 for 'ortho6d'.
        w_first (bool): Whether the quaternion representation uses (qw, qx, qy, qz) format.

    Returns:
        torch.Tensor: A tensor of shape (..., 9) representing the pose in position and 6-DOF rotation representation.
    """
    representation_from = 'quat' if pose.shape[-1] == 7 else 'ortho6d' if pose.shape[-1] == 9 else 'T_mat' if (pose.shape[-1] == 4) and (pose.shape[-2] == 4) else None

    if representation_from == 'ortho6d':
        return _pose_ortho6d_to_pose_quat(pose, w_first=w_first), representation_from
    elif representation_from == 'T_mat':
        return _T_to_pose_quat(pose, w_first=w_first), representation_from
    elif representation_from == 'quat':
        return pose, representation_from
    else:
        raise ValueError(f"Unsupported representation_from. Supported: 'quat', 'T_matrix', 'ortho6d'.")

def _convert_pose_quat_to_pose_any(pose_quat, representation_to, w_first=True):
    """
    Convert a batch of poses from position and quaternion to any supported representation.
    
    Args:
        pose_quat (torch.Tensor): A tensor of shape (..., 9) representing the pose in position and 6-DOF rotation representation.
        representation_to (str): The target representation: 'quat', 'ortho6d', or 'T_mat'.
        w_first (bool): Whether the quaternion representation uses (qw, qx, qy, qz) format.

    Returns:
        torch.Tensor: A tensor of shape (..., D) where D is 7 for 'quat', 9 for 'ortho6d', or (4,4) for 'T_mat'.
    """
    if representation_to == 'ortho6d':
        return _pose_quat_to_pose_ortho6d(pose_quat, w_first=w_first)
    elif representation_to == 'T_mat':
        return _pose_quat_to_T(pose_quat, w_first=w_first)
    elif representation_to == 'quat':
        return pose_quat
    else:
        raise ValueError(f"Unsupported representation_to. Supported: 'quat', 'T_matrix', 'ortho6d'.")

# quaternion math ops
def _quaternion_multiply(q1, q2):
        """
        Multiply two quaternions together (q1 * q2)
        
        Args:
            q1: First quaternion of shape [..., 4] in (qw, qx, qy, qz) format
            q2: Second quaternion of shape [..., 4] in (qw, qx, qy, qz) format
            
        Returns:
            Quaternion product of shape [..., 4]
        """
        w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
        w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2
        
        return torch.cat([w, x, y, z], dim=-1)

def _quaternion_difference(q1, q2):
    """
    Calculate the quaternion difference q_diff = q2 * q1^(-1)
    
    Args:
        q1: First quaternion of shape [..., 4] in (qw, qx, qy, qz) format
        q2: Second quaternion of shape [..., 4] in (qw, qx, qy, qz) format
        
    Returns:
        Quaternion difference of shape [..., 4]
    """
    # For unit quaternions, inverse is just the conjugate
    q1_inv = torch.cat([q1[..., 0:1], -q1[..., 1:4]], dim=-1)
    
    # Quaternion multiplication q2 * q1_inv
    w1, x1, y1, z1 = q1_inv[..., 0:1], q1_inv[..., 1:2], q1_inv[..., 2:3], q1_inv[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    q_diff = torch.cat([w, x, y, z], dim=-1)
    
    # Normalize to fix numerical errors
    q_diff = q_diff / torch.norm(q_diff, dim=-1, keepdim=True)
    
    return q_diff

##########################
# --- PUBLIC METHODS --- #
##########################
# twist math ops
def compute_twist_between_poses(pose1, pose2=None, dt=1.0, relative_pose=None):
    """
    Compute the twist (spatial velocity) between two sets of poses.
    
    Args:
        pose1: First pose as tensor of shape [..., 7], [..., 9], or [..., 4, 4]
        pose2: Second pose as tensor of shape [..., 7], [..., 9], or [..., 4, 4] or None. If None, 
                convert pose1 into a twist relative to the world frame (identity).
        dt: Time difference between poses (default=1.0)
        relative_pose: pose that the twist is relative to (default: None) of 
                shape [..., 7], [..., 9], or [..., 4, 4]. If None, twist is relative
                to world frame
        
    Returns:
        Twist vector of shape [..., 6] containing [vx, vy, vz, ωx, ωy, ωz]
    """

    # Handle case where pose2 is None (convert pose1 to twist from identity)
    if pose2 is None:
        # Create identity pose with same shape as pose1
        batch_shape = pose1.shape[:-1]
        identity_pose = torch.zeros((*batch_shape, 7), device=pose1.device, dtype=pose1.dtype)
        identity_pose[..., 3] = 1.0  # Set qw = 1 for identity quaternion
        pose2 = pose1
        pose1 = identity_pose

    # Set up relative_pose (default to identity/world frame)
    if relative_pose is None:
        # Create identity pose with same shape as pose1
        batch_shape = pose1.shape[:-1]
        relative_pose = torch.zeros((*batch_shape, 7), device=pose1.device, dtype=pose1.dtype)
        relative_pose[..., 3] = 1.0  # Set qw = 1 for identity quaternion
    else:
        # put the relative pose on the correct device and dtype
        relative_pose = relative_pose.to(pose1.device).to(pose1.dtype)

    # convert poses to position and quaternion (x, y, z, qw, qx, qy, qz)
    pose1, _ = _convert_pose_any_to_pose_quat(pose1)
    pose2, _ = _convert_pose_any_to_pose_quat(pose2)
    relative_pose, _ = _convert_pose_any_to_pose_quat(relative_pose)

    # check that pose1 and pose2 batch dimensions are compatible
    if pose1.shape[:-1] != pose2.shape[:-1]:
        raise ValueError(f"Pose1 batch dim {pose1.shape[:-1]} and Pose2 batch dim {pose2.shape[:-1]} are not compatible.")

    # Extract positions and quaternions
    pos1, quat1 = pose1[..., :3], pose1[..., 3:7]
    pos2, quat2 = pose2[..., :3], pose2[..., 3:7]
    rel_pos, rel_quat = relative_pose[..., :3], relative_pose[..., 3:7]
    
    # Compute linear velocity in world frame (position difference)
    linear_vel_world = (pos2 - pos1) / dt
    
    # Compute quaternion difference (relative rotation)
    quat_diff = _quaternion_difference(quat1, quat2)
    
    # Convert quaternion difference to angular velocity in world frame
    qw = quat_diff[..., 0]
    qxyz = quat_diff[..., 1:4]
    
    # Compute rotation angle from quaternion
    angle = 2 * torch.acos(torch.clamp(qw, min=-1.0, max=1.0))
    
    # Compute rotation axis (handling edge cases)
    norm_qxyz = torch.norm(qxyz, dim=-1, keepdim=True)
    safe_norm_qxyz = torch.where(norm_qxyz > 1e-6, norm_qxyz, torch.ones_like(norm_qxyz))
    axis = qxyz / safe_norm_qxyz
    
    # Angular velocity = axis * angle / dt (in world frame)
    angular_vel_world = axis * angle.unsqueeze(-1) / dt
    
    # Transform twist to relative frame using inverse rotation of relative_pose
    # Get conjugate (inverse) of relative quaternion
    rel_quat_inv = torch.cat([rel_quat[..., 0:1], -rel_quat[..., 1:4]], dim=-1)
    rel_rot_mat = _quat_to_rot_mat(rel_quat_inv, w_first=True)
    
    # Rotate velocities into relative frame
    linear_vel = torch.matmul(rel_rot_mat, linear_vel_world.unsqueeze(-1)).squeeze(-1)
    angular_vel = torch.matmul(rel_rot_mat, angular_vel_world.unsqueeze(-1)).squeeze(-1)
    
    # Combine linear and angular velocity into twist
    twist = torch.cat([linear_vel, angular_vel], dim=-1)
    
    return twist

def sample_random_twist(batch_size=1, mu=[0,0,0,0,0,0], sigma=[1,1,1,1,1,1], device='cpu', dtype=torch.float32):
    """
    Sample a random twist vector.
    
    Args:
        batch_size: Shape of batch dimensions - can be int or tuple/list for multi-dimensional batches
        mu: Mean of the twist distribution [vx, vy, vz, ωx, ωy, ωz]
        sigma: Standard deviation of the twist distribution [vx, vy, vz, ωx, ωy, ωz]
        device: Device to create the tensor on
        dtype: Data type of the tensor
        
    Returns:
        Twist tensor of shape [*batch_size, 6] containing [vx, vy, vz, ωx, ωy, ωz]
    """
    # Convert batch_size to tuple if it's an integer
    if isinstance(batch_size, int):
        batch_size = (batch_size,)
    
    # Expand mean and std to include batch dimensions
    linear_vel = torch.normal(mean=torch.tensor(mu[:3], device=device, dtype=dtype).expand(*batch_size, -1),
                              std=torch.tensor(sigma[:3], device=device, dtype=dtype).expand(*batch_size, -1))
    angular_vel = torch.normal(mean=torch.tensor(mu[3:], device=device, dtype=dtype).expand(*batch_size, -1),
                               std=torch.tensor(sigma[3:], device=device, dtype=dtype).expand(*batch_size, -1))
    twist = torch.cat([linear_vel, angular_vel], dim=-1)
    return twist

def add_twist_to_pose(pose, twist, dt):
    """
    Add a twist over time dt to a pose
    
    Args:
        pose: Tensor of shape [..., 7] (x, y, z, qw, qx, qy, qz)
        twist: Tensor of shape [..., 6] (vx, vy, vz, ωx, ωy, ωz)
        dt: Tensor of shape [..., 1] Time period to apply twist for
        
    Returns:
        Updated pose of shape [..., 7]
    """
   
    # convert pose to position and quaternion (x, y, z, qw, qx, qy, qz)
    pose, pose_rep = _convert_pose_any_to_pose_quat(pose)

    # Ensure dt has the correct shape [..., 1]
    if dt.dim() == 0:  # scalar tensor
        dt = dt.unsqueeze(-1)
    elif dt.shape[-1] != 1:
        dt = dt.unsqueeze(-1)
    
    # Broadcast dt to match batch dimensions if needed
    if dt.shape[:-1] != pose.shape[:-1]:
        target_shape = pose.shape[:-1] + (1,)
        dt = dt.expand(target_shape)

    # check that twist and pose batch dimensions are compatible
    if pose.shape[:-1] != twist.shape[:-1]:
        raise ValueError(f"Pose batch dim {pose.shape[:-1]} and twist batch dim {twist.shape[:-1]} are not compatible.")

    # Extract components
    position = pose[..., :3]
    quaternion = pose[..., 3:7]
    linear_vel = twist[..., :3]
    angular_vel = twist[..., 3:6]
    
    # 1. Update position (simple integration)
    new_position = position + linear_vel * dt
    
    # 2. Update orientation using exponential map
    # Calculate the rotation angle from angular velocity
    angle = torch.norm(angular_vel, dim=-1, keepdim=True) * dt
    
    # Create a unit rotation axis (handling zero angular velocity case)
    axis_norm = torch.norm(angular_vel, dim=-1, keepdim=True)
    axis = torch.where(
        axis_norm > 1e-6,
        angular_vel / axis_norm,
        torch.tensor([1.0, 0.0, 0.0], device=pose.device).expand_as(angular_vel)
    )
    
    # Create the rotation quaternion (from axis-angle)
    half_angle = angle * 0.5
    sin_half = torch.sin(half_angle)
    
    rot_quat = torch.cat([
        torch.cos(half_angle),
        axis * sin_half
    ], dim=-1)
    
    # Apply the rotation using quaternion multiplication
    new_quaternion = _quaternion_multiply(quaternion, rot_quat)
    
    # Normalize the resulting quaternion
    new_quaternion = new_quaternion / torch.norm(new_quaternion, dim=-1, keepdim=True)
    
    # Combine into new pose
    new_pose = torch.cat([new_position, new_quaternion], dim=-1)

    # convert pose back to original representation
    new_pose = _convert_pose_quat_to_pose_any(new_pose, representation_to=pose_rep)
    
    return new_pose

def convert_twist_to_pose(twist, dt=1.0, relative_pose=None, return_representation='quat'):
    """
    Convert a twist vector to a pose after applying the twist over time dt using the skew exponential map.
    
    Args:
        twist: Tensor of shape [..., 6] (vx, vy, vz, ωx, ωy, ωz)
        dt: Time period to apply twist for (default=1.0)
        relative_pose: pose that the twist is relative to (default: None) of 
                shape [..., 7], [..., 9], or [..., 4, 4]. If None, twist is relative
                to world frame
        return_representation: Desired output pose representation: 'quat', 'ortho6d', or 'T_mat' (default='quat')
                
    Returns:
        Pose tensor of shape [..., 7] (x, y, z, qw, qx, qy, qz)
    """
    
    # Ensure dt has the correct shape [..., 1]
    if isinstance(dt, (int, float)):
        dt = torch.tensor(dt, device=twist.device, dtype=twist.dtype)
    if dt.dim() == 0:  # scalar tensor
        dt = dt.unsqueeze(-1)
    elif dt.shape[-1] != 1:
        dt = dt.unsqueeze(-1)
    
    # Broadcast dt to match batch dimensions if needed
    if dt.shape[:-1] != twist.shape[:-1]:
        target_shape = twist.shape[:-1] + (1,)
        dt = dt.expand(target_shape)
    
    # Get the twist in world frame
    twist_world = twist
    
    # If relative_pose is provided, transform twist from relative frame to world frame
    if relative_pose is not None:
        # Convert relative_pose to quaternion representation
        relative_pose, _ = _convert_pose_any_to_pose_quat(relative_pose)
        rel_quat = relative_pose[..., 3:7]
        
        # Get rotation matrix from quaternion (not inverse, we want to rotate FROM relative frame TO world frame)
        rel_rot_mat = _quat_to_rot_mat(rel_quat, w_first=True)
        
        # Extract linear and angular velocities
        linear_vel_rel = twist[..., :3]
        angular_vel_rel = twist[..., 3:6]
        
        # Rotate velocities from relative frame to world frame
        linear_vel_world = torch.matmul(rel_rot_mat, linear_vel_rel.unsqueeze(-1)).squeeze(-1)
        angular_vel_world = torch.matmul(rel_rot_mat, angular_vel_rel.unsqueeze(-1)).squeeze(-1)
        
        # Combine into world-frame twist
        twist_world = torch.cat([linear_vel_world, angular_vel_world], dim=-1)
    
    # Create identity pose to start from
    batch_shape = twist.shape[:-1]
    identity_pose = torch.zeros((*batch_shape, 7), device=twist.device, dtype=twist.dtype)
    identity_pose[..., 3] = 1.0  # Set qw = 1 for identity quaternion
    
    # Apply the twist to the identity pose
    result_pose = add_twist_to_pose(identity_pose, twist_world, dt)
    
    # Convert to desired representation
    result_pose = _convert_pose_quat_to_pose_any(result_pose, representation_to=return_representation)
    
    return result_pose


#################
# --- TESTS --- #
#################

def test_twist_operations():
    """Test twist computation and application operations"""
    print("Testing twist operations...")
    
    # Test 1: Basic twist computation between two poses
    print("\n1. Testing basic twist computation...")
    pose1 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Identity
    pose2 = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])  # Translation only
    dt = 1.0
    
    twist = compute_twist_between_poses(pose1, pose2, dt=dt)
    print(f"   Pose1: {pose1}")
    print(f"   Pose2: {pose2}")
    print(f"   Computed twist: {twist}")
    print(f"   Expected linear vel: [1, 2, 3], Got: {twist[:3]}")
    assert torch.allclose(twist[:3], torch.tensor([1.0, 2.0, 3.0]), atol=1e-5), "Linear velocity mismatch"
    assert torch.allclose(twist[3:], torch.zeros(3), atol=1e-5), "Angular velocity should be zero"
    print("   ✓ Basic translation test passed")
    
    # Test 2: Rotation only (90 degree rotation around Z axis)
    print("\n2. Testing rotation-only twist...")
    import math
    angle = math.pi / 2  # 90 degrees
    quat_z_90 = torch.tensor([math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)])
    pose1 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    pose2 = torch.cat([torch.zeros(3), quat_z_90])
    
    twist = compute_twist_between_poses(pose1, pose2, dt=1.0)
    print(f"   Rotation twist: {twist}")
    print(f"   Angular velocity around Z: {twist[5]}")
    assert torch.allclose(twist[:3], torch.zeros(3), atol=1e-5), "Linear velocity should be zero"
    assert torch.abs(twist[5] - math.pi/2) < 1e-4, "Angular velocity should be π/2 around Z"
    print("   ✓ Rotation test passed")
    
    # Test 3: Invertibility - add_twist_to_pose and compute_twist should be inverses
    print("\n3. Testing invertibility...")
    pose_start = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    twist_test = torch.tensor([0.5, -0.3, 0.7, 0.1, -0.2, 0.15])
    dt = 0.5
    
    pose_end = add_twist_to_pose(pose_start, twist_test, torch.tensor(dt))
    twist_recovered = compute_twist_between_poses(pose_start, pose_end, dt=dt)
    
    print(f"   Original twist: {twist_test}")
    print(f"   Recovered twist: {twist_recovered}")
    print(f"   Difference: {torch.abs(twist_test - twist_recovered)}")
    assert torch.allclose(twist_test, twist_recovered, atol=1e-4), "Twist recovery failed"
    print("   ✓ Invertibility test passed")
    
    # Test 4: Batch operations
    print("\n4. Testing batch operations...")
    batch_size = 5
    pose1_batch = torch.zeros(batch_size, 7)
    pose1_batch[..., 3] = 1.0  # Set qw = 1
    pose2_batch = torch.rand(batch_size, 7)
    pose2_batch[..., 3:] = pose2_batch[..., 3:] / torch.norm(pose2_batch[..., 3:], dim=-1, keepdim=True)
    
    twist_batch = compute_twist_between_poses(pose1_batch, pose2_batch, dt=1.0)
    print(f"   Batch shape: {twist_batch.shape}")
    assert twist_batch.shape == (batch_size, 6), "Batch shape mismatch"
    print("   ✓ Batch operations test passed")
    
    # Test 5: pose2=None (convert pose to twist from identity)
    print("\n5. Testing pose2=None (pose to twist)...")
    pose = torch.tensor([2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0])
    twist_from_identity = compute_twist_between_poses(pose, pose2=None, dt=1.0)
    print(f"   Pose: {pose[:3]}")
    print(f"   Twist from identity: {twist_from_identity}")
    assert torch.allclose(twist_from_identity[:3], pose[:3], atol=1e-5), "Position should match linear velocity"
    print("   ✓ pose2=None test passed")
    
    # Test 6: Relative pose frame transformation
    print("\n6. Testing relative pose transformation...")
    # Create a pose offset by (1,0,0) and rotated 45 degrees around Z
    angle = math.pi / 4
    rel_pose = torch.tensor([1.0, 0.0, 0.0, math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)])
    
    pose1 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    pose2 = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    
    # Twist in world frame
    twist_world = compute_twist_between_poses(pose1, pose2, dt=1.0, relative_pose=None)
    # Twist in relative frame
    twist_rel = compute_twist_between_poses(pose1, pose2, dt=1.0, relative_pose=rel_pose)
    
    print(f"   Twist (world frame): {twist_world}")
    print(f"   Twist (relative frame): {twist_rel}")
    assert not torch.allclose(twist_world, twist_rel, atol=1e-4), "Twists should differ in different frames"
    print("   ✓ Relative frame transformation test passed")
    
    # Test 7: Different pose representations (ortho6d and T_mat)
    print("\n7. Testing different pose representations...")
    pose_quat = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    pose_ortho6d = _pose_quat_to_pose_ortho6d(pose_quat)
    pose_T = _pose_quat_to_T(pose_quat)
    
    twist1 = compute_twist_between_poses(pose_quat, pose_quat + torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]), dt=1.0)
    twist2 = compute_twist_between_poses(pose_ortho6d, _pose_quat_to_pose_ortho6d(pose_quat + torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])), dt=1.0)
    twist3 = compute_twist_between_poses(pose_T, _pose_quat_to_T(pose_quat + torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])), dt=1.0)
    
    print(f"   Twist from quat: {twist1}")
    print(f"   Twist from ortho6d: {twist2}")
    print(f"   Twist from T_mat: {twist3}")
    assert torch.allclose(twist1, twist2, atol=1e-4), "Twist from ortho6d should match quat"
    assert torch.allclose(twist1, twist3, atol=1e-4), "Twist from T_mat should match quat"
    print("   ✓ Different representations test passed")
    
    # Test 8: Add twist with dt scaling
    print("\n8. Testing add_twist_to_pose with different dt values...")
    pose_base = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    twist = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    
    pose_dt1 = add_twist_to_pose(pose_base, twist, torch.tensor(1.0))
    pose_dt2 = add_twist_to_pose(pose_base, twist, torch.tensor(2.0))
    pose_dt05 = add_twist_to_pose(pose_base, twist, torch.tensor(0.5))
    
    print(f"   Base pose: {pose_base[:3]}")
    print(f"   After dt=1.0: {pose_dt1[:3]}")
    print(f"   After dt=2.0: {pose_dt2[:3]}")
    print(f"   After dt=0.5: {pose_dt05[:3]}")
    assert torch.allclose(pose_dt1[:3], torch.tensor([1.0, 0.0, 0.0]), atol=1e-4), "Position after dt=1 incorrect"
    assert torch.allclose(pose_dt2[:3], torch.tensor([2.0, 0.0, 0.0]), atol=1e-4), "Position after dt=2 incorrect"
    print("   ✓ dt scaling test passed")
    
    # Test 9: sample_random_twist
    print("\n9. Testing sample_random_twist...")
    random_twists = sample_random_twist(batch_size=10, mu=[0,0,0,0,0,0], sigma=[1,1,1,1,1,1])
    print(f"   Random twist shape: {random_twists.shape}")
    print(f"   Mean: {random_twists.mean(dim=0)}")
    print(f"   Std: {random_twists.std(dim=0)}")
    assert random_twists.shape == (10, 6), "Random twist shape mismatch"
    assert torch.abs(random_twists.mean()) < 1.0, "Mean should be close to zero"
    print("   ✓ Random twist sampling test passed")
    
    # Test 10: Round-trip test with rotation
    print("\n10. Testing round-trip with complex rotation...")
    pose_start = torch.tensor([1.0, 2.0, 3.0, 0.9239, 0.3827, 0.0, 0.0])  # 45 deg around X
    twist = torch.tensor([0.5, 0.3, -0.2, 0.1, 0.2, 0.3])
    dt = 1.5
    
    pose_end = add_twist_to_pose(pose_start, twist, torch.tensor(dt))
    twist_recovered = compute_twist_between_poses(pose_start, pose_end, dt=dt)
    pose_roundtrip = add_twist_to_pose(pose_start, twist_recovered, torch.tensor(dt))
    
    print(f"   Start pose: {pose_start}")
    print(f"   End pose: {pose_end}")
    print(f"   Round-trip pose: {pose_roundtrip}")
    print(f"   Pose difference: {torch.abs(pose_end - pose_roundtrip)}")
    assert torch.allclose(pose_end, pose_roundtrip, atol=1e-3), "Round-trip failed"
    print("   ✓ Round-trip test passed")
    
    # Test 11: convert_twist_to_pose basic functionality
    print("\n11. Testing convert_twist_to_pose...")
    twist = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    pose_from_twist = convert_twist_to_pose(twist, dt=1.0)
    print(f"   Twist: {twist}")
    print(f"   Resulting pose: {pose_from_twist}")
    assert torch.allclose(pose_from_twist[:3], torch.tensor([1.0, 2.0, 3.0]), atol=1e-4), "Position mismatch"
    assert torch.allclose(pose_from_twist[3:], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-4), "Should be identity rotation"
    print("   ✓ Basic convert_twist_to_pose test passed")
    
    # Test 12: convert_twist_to_pose with rotation
    print("\n12. Testing convert_twist_to_pose with rotation...")
    twist_rot = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, math.pi/2])  # 90 deg rotation around Z
    pose_rot = convert_twist_to_pose(twist_rot, dt=1.0)
    print(f"   Twist (rotation only): {twist_rot}")
    print(f"   Resulting pose: {pose_rot}")
    # Should have zero position and quaternion for 90 deg Z rotation
    assert torch.allclose(pose_rot[:3], torch.zeros(3), atol=1e-4), "Position should be zero"
    expected_quat = torch.tensor([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)])
    assert torch.allclose(pose_rot[3:], expected_quat, atol=1e-4), "Quaternion mismatch"
    print("   ✓ Rotation test passed")
    
    # Test 13: convert_twist_to_pose with relative_pose
    print("\n13. Testing convert_twist_to_pose with relative_pose...")
    # Twist in a rotated frame should give different result
    angle = math.pi / 4
    rel_pose = torch.tensor([0.0, 0.0, 0.0, math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)])
    twist_local = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Move 1 unit in local X
    
    pose_world = convert_twist_to_pose(twist_local, dt=1.0, relative_pose=None)
    pose_rel = convert_twist_to_pose(twist_local, dt=1.0, relative_pose=rel_pose)
    
    print(f"   Twist (local): {twist_local}")
    print(f"   Pose (world frame): {pose_world[:3]}")
    print(f"   Pose (relative frame): {pose_rel[:3]}")
    assert not torch.allclose(pose_world[:3], pose_rel[:3], atol=1e-4), "Poses should differ"
    print("   ✓ Relative frame test passed")
    
    # Test 14: convert_twist_to_pose with different return representations
    print("\n14. Testing convert_twist_to_pose with different representations...")
    twist = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    
    pose_quat = convert_twist_to_pose(twist, dt=1.0, return_representation='quat')
    pose_ortho6d = convert_twist_to_pose(twist, dt=1.0, return_representation='ortho6d')
    pose_T = convert_twist_to_pose(twist, dt=1.0, return_representation='T_mat')
    
    print(f"   Quat shape: {pose_quat.shape}")
    print(f"   Ortho6d shape: {pose_ortho6d.shape}")
    print(f"   T_mat shape: {pose_T.shape}")
    assert pose_quat.shape[-1] == 7, "Quat should be shape 7"
    assert pose_ortho6d.shape[-1] == 9, "Ortho6d should be shape 9"
    assert pose_T.shape[-2:] == (4, 4), "T_mat should be shape (4,4)"
    
    # Verify they represent the same pose
    pose_quat_from_ortho = _pose_ortho6d_to_pose_quat(pose_ortho6d)
    pose_quat_from_T = _T_to_pose_quat(pose_T)
    assert torch.allclose(pose_quat, pose_quat_from_ortho, atol=1e-4), "Ortho6d conversion mismatch"
    assert torch.allclose(pose_quat, pose_quat_from_T, atol=1e-4), "T_mat conversion mismatch"
    print("   ✓ Different representation test passed")
    
    # Test 15: Invertibility with convert_twist_to_pose
    print("\n15. Testing invertibility: twist -> pose -> twist...")
    twist_orig = torch.tensor([0.5, -0.3, 0.7, 0.1, -0.2, 0.15])
    dt = 0.8
    
    pose_from_twist = convert_twist_to_pose(twist_orig, dt=dt)
    twist_back = compute_twist_between_poses(torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), pose_from_twist, dt=dt)
    
    print(f"   Original twist: {twist_orig}")
    print(f"   Recovered twist: {twist_back}")
    print(f"   Difference: {torch.abs(twist_orig - twist_back)}")
    assert torch.allclose(twist_orig, twist_back, atol=1e-4), "Twist recovery failed"
    print("   ✓ Invertibility test passed")
    
    print("\n✅ All twist operation tests passed!")

if __name__ == "__main__":
    # test_twist_operations()

    # start pose distribution at the origin
    start_poses = sample_random_twist(batch_size=3, mu=[0,0,0,0,0,0], sigma=[1,1,1,1,1,1])
    print("start_poses (as twists):\n", start_poses)
    start_poses_ortho6d = convert_twist_to_pose(start_poses, dt=1.0, return_representation='ortho6d')
    print("start_poses (as ortho6d poses):\n", start_poses_ortho6d)
    start_poses_T = convert_twist_to_pose(start_poses, dt=1.0, return_representation='T_mat')
    print("start_poses (as T_mat poses):\n", start_poses_T)

    # goal pose distribution at different position with same orientation
    goal_poses = sample_random_twist(batch_size=3, mu=[5,5,5,0,0,0], sigma=[0.1,0.1,0.1,0.1,0.1,0.1])
    print("goal_poses (as twists):\n", goal_poses)
    goal_poses_ortho6d = convert_twist_to_pose(goal_poses, dt=1.0, return_representation='ortho6d')
    print("goal_poses (as ortho6d poses):\n", goal_poses_ortho6d)
    goal_poses_T = convert_twist_to_pose(goal_poses, dt=1.0, return_representation='T_mat')
    print("goal_poses (as T_mat poses):\n", goal_poses_T)





