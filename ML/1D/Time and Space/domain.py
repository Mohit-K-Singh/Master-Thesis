import torch
import numpy as np


def sample_space_time_points(n_interior=10000, n_boundary=1000, n_initial=2000, x_range=(-5.0, 5.0), t_range=(0.0, 7)):
    """
    Samples space-time points for training a PINN for the time-dependent Schrödinger equation.
    """
    a, b = x_range
    t0, t1 = t_range

    # 1. Interior points: for PDE residual
    x_interior = torch.rand(n_interior, 1) * (b - a) + a
    t_interior = torch.rand(n_interior, 1) * (t1 - t0) + t0

    # 2. Initial condition points (t=0)
    x_initial = torch.rand(n_initial, 1) * (b - a) + a
    t_initial = torch.zeros_like(x_initial)

    # 3. Boundary condition points: sample at x = a and x = b for multiple t
    t_boundary = torch.rand(2*n_boundary,1) * (t1 - t0) + t0
    #x_boundary_left = torch.full_like(t_boundary, a)
    #x_boundary_right = torch.full_like(t_boundary, b)
    x_boundary_left = -7 + 2 * torch.rand(n_boundary,1)
    x_boundary_right = 5 + 2 * torch.rand(n_boundary,1)
    x_boundary = torch.cat([x_boundary_left, x_boundary_right], dim=0)
    # Return all sampled points
    #interior = torch.cat([x_interior, t_interior], dim=1)
    #initial = torch.cat([x_initial, t_initial], dim=1)
    #boundary = torch.cat([torch.cat([x_boundary_left, x_boundary_right], dim=0),torch.cat([t_boundary, t_boundary], dim=0)], dim=1)

    return x_interior,t_interior, x_initial, t_initial, x_boundary, t_boundary

