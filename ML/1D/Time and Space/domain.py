import torch
import numpy as np


def sample_space_time_points(device, n_interior=5000, n_boundary=700, n_initial=3000, x_range=(-5.0, 5.0), t_range=(0.0, 1)):
    """
    Samples space-time points for training a PINN for the time-dependent Schrödinger equation.
    """
    a, b = x_range
    t0, t1 = t_range

    # 1. Interior points: for PDE residual
    x_interior,_ = torch.sort(torch.rand(n_interior, 1, device=device) * (b - a) + a, dim=0)
    t_interior,_ = torch.sort(torch.rand(n_interior, 1, device=device) * (t1 - t0) + t0, dim=0)

    # 2. Initial condition points (t=0)
    x_initial,_ = torch.sort(torch.rand(n_initial, 1, device=device) * (b - a) + a, dim=0)
    t_initial = torch.zeros_like(x_initial)

    # 3. Boundary condition points: sample at x = a and x = b for multiple t
    t_boundary, _ = torch.sort(torch.rand(2*n_boundary,1, device=device) * (t1 - t0) + t0, dim=0)
    x_boundary_left = -7 + 2 * torch.rand(n_boundary,1, device=device)
    x_boundary_right = 5 + 2 * torch.rand(n_boundary,1, device=device)
    x_boundary,_ = torch.sort(torch.cat([x_boundary_left, x_boundary_right], dim=0), dim=0)


    return x_interior,t_interior, x_initial, t_initial, x_boundary, t_boundary

