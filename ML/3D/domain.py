import torch

def sample_points2(n_interior = 10000, n_boundary = 700):
    """
    Sample from domain [-2.5, 2.5]³ via torch.rand().
    """
    a, b = -2.5, 2.5
    interior = torch.rand(n_interior, 3) * (b-a) + a

    lin = (torch.rand(n_boundary, 2) * (b-a) + a)
    bottom = torch.cat([lin, torch.full((lin.shape[0],1), a)], dim = 1)
    top = torch.cat([lin, torch.full((lin.shape[0],1), b)], dim=1)       
    
    # Sample a new face
    lin = (torch.rand(n_boundary, 2) * (b-a) + a)
    back = torch.cat([torch.full((lin.shape[0],1), a), lin], dim =1)
    front = torch.cat([torch.full((lin.shape[0],1), b), lin], dim =1)
    
    lin = (torch.rand(n_boundary, 2) * (b-a) + a)
    # Create new columns to be inserted for the right and left face
    boundary_dim_plus = torch.full((lin.shape[0],1), b)
    boundary_dim_minus = torch.full((lin.shape[0],1), a)

    left = torch.cat([lin[:,[0]], boundary_dim_minus, lin[:, [1]]] , dim = 1)
    right = torch.cat([lin[:,[0]], boundary_dim_plus, lin[:, [1]]] , dim = 1)
    # Bring it all together 
    boundary = torch.cat([bottom, top, back, front, left, right], dim=0)
 
    return interior, boundary

def sample_points(device, n_interior=10000, n_boundary=1000):
    a, b = -3.5, 3.5

    # Interior points (unchanged)
    interior = torch.rand(n_interior, 3,device=device) * (b - a) + a

    # Generate all boundary points at once (n_boundary × 6 × 2)
    lin = torch.rand(n_boundary * 6, 2, device=device) * (b - a) + a

    # Split into 6 faces (each n_boundary × 2)
    bottom = torch.cat([lin[:n_boundary], torch.full((n_boundary, 1), a, device=device)], dim=1)
    top = torch.cat([lin[n_boundary:2*n_boundary], torch.full((n_boundary, 1), b, device=device)], dim=1)

    back = torch.cat([torch.full((n_boundary, 1), a,device=device), lin[2*n_boundary:3*n_boundary]], dim=1)
    front = torch.cat([torch.full((n_boundary, 1), b,device=device), lin[3*n_boundary:4*n_boundary]], dim=1)

    left = torch.stack([
        lin[4*n_boundary:5*n_boundary, 0],
        torch.full((n_boundary,), a, device=device),
        lin[4*n_boundary:5*n_boundary, 1]
    ], dim=1)

    right = torch.stack([
        lin[5*n_boundary:, 0],
        torch.full((n_boundary,), b, device=device),
        lin[5*n_boundary:, 1]
    ], dim=1)

    boundary = torch.cat([bottom, top, back, front, left, right], dim=0)
    return interior, boundary
