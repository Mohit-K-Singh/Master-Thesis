import torch


def sample_points(device,n_interior = 3000, n_boundary = 1000):
    """
    Sample points from the boundary and interior uniformly via torch.rand()
    The domain is [-4,4]²
    """
    a, b = -4.0, 4.0

    # Interior sampling
    interior = torch.rand(n_interior, 2,device=device) * (b-a) + a

    # Boundary sampling
    lin = torch.rand(n_boundary*4, 1,device=device) * (b-a) + a # shape [200,1]

    # Create bottom,top,left and right side ..
    bottom = torch.cat([lin[:n_boundary], torch.full((n_boundary,1), a,device=device)], dim=1)  
    top = torch.cat([lin[n_boundary:2*n_boundary], torch.full((n_boundary, 1), b,device=device)], dim=1)
    left = torch.cat([torch.full((n_boundary, 1), a,device=device), lin[2*n_boundary:3*n_boundary]], dim=1)
    right = torch.cat([torch.full((n_boundary, 1), b,device=device), lin[3*n_boundary:]], dim=1)
    # ... and connect them
    boundary = torch.cat([bottom, top, left, right], dim=0)
    
    return interior, boundary

