import torch


def add_velocity_to_state(state, velocity, dt):
    """
    Euclidean analog of `add_twist_to_pose`. Performs a single Euler step.

    Args:
        state: Tensor of shape [..., D]
        velocity: Tensor of shape [..., D]
        dt: scalar tensor, tensor of shape [..., 1], or shape broadcastable to state.

    Returns:
        Updated state of shape [..., D].
    """
    if isinstance(dt, torch.Tensor):
        if dt.dim() == 0:
            dt = dt.unsqueeze(-1)
        elif dt.shape[-1] != 1 and dt.shape != state.shape:
            dt = dt.unsqueeze(-1)

    if state.shape != velocity.shape:
        raise ValueError(
            f"State shape {state.shape} and velocity shape {velocity.shape} must match."
        )

    return state + velocity * dt


def compute_velocity_between_states(start, goal, dt=1.0):
    """
    Euclidean analog of `compute_twist_between_poses`. The "velocity" that, applied
    for `dt`, carries `start` to `goal` under the Euclidean Euler map is simply
    `(goal - start) / dt`.

    Args:
        start: Tensor of shape [..., D]
        goal: Tensor of shape [..., D]
        dt: scalar (default 1.0). Provided for API parity with the SE(3) version.

    Returns:
        Velocity tensor of shape [..., D].
    """
    if start.shape != goal.shape:
        raise ValueError(
            f"Start shape {start.shape} and goal shape {goal.shape} must match."
        )
    return (goal - start) / dt
