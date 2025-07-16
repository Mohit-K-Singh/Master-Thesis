import torch

def sample_points(device, n_interior = 7000, n_boundary = 200):
    """
    Sample points in the domain [-7,7]
    """
    a, b = -5.0, 5.0
    interior = torch.rand(n_interior, 1, device=device) * (b-a) + a
    
    boundary_left = -7 + 2 * torch.rand(n_boundary,1, device=device)
    boundary_right = 5 + 2 * torch.rand(n_boundary,1, device=device)
    boundary = torch.cat([boundary_left, boundary_right], dim= 0)
    return interior, boundary
