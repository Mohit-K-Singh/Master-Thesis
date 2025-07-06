import torch


def sample_points(n_interior = 3000, n_boundary = 200):
    """
    Sample points from the boundary and interior uniformly via torch.rand()
    The domain is [-5,5]²
    """
    a, b = -5.0, 5.0

    # Interior sampling
    interior = torch.rand(n_interior, 2) * (b-a) + a

    # Boundary sampling
    lin = torch.rand(n_boundary*4, 1) * (b-a) + a # shape [200,1]

    # Create bottom,top,left and right side ..
    bottom = torch.cat([lin[:n_boundary], torch.full((n_boundary,1), a)], dim=1)  
    top = torch.cat([lin[n_boundary:2*n_boundary], torch.full((n_boundary, 1), b)], dim=1)
    left = torch.cat([torch.full((n_boundary, 1), a), lin[2*n_boundary:3*n_boundary]], dim=1)
    right = torch.cat([torch.full((n_boundary, 1), b), lin[3*n_boundary:]], dim=1)
    # ... and connect them
    boundary = torch.cat([bottom, top, left, right], dim=0)
    
    return interior, boundary

