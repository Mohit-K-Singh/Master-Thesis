import torch
import numpy as np

def sample_space_time_points(device ,n_interior=6000, n_boundary=400, n_initial=2000, x_range=(-3.5, 3.5), t_range=(0.0, 1)):

    a, b = x_range
    t0, t1 = t_range

    x_interior = torch.rand(n_interior, 2 ,device=device) * (b-a) + a
    t_interior = torch.rand(n_interior,1,device=device) * (t1 - t0) + t0


    x_initial = torch.rand(n_initial,2,device=device) * (b-a) + a
    t_initial = torch.zeros_like(x_initial[:,0], device=device).unsqueeze(-1)

    #x_initial = torch.linspace(-4,4, n_initial).to(device)
    #t_initial = torch.zeros_like(x_initial).to(device)

    lin= torch.rand(n_boundary*4, 1, device=device) * (b-a) + a
    bottom = torch.cat([lin[:n_boundary], torch.full((n_boundary,1), a, device=device)], dim=1)  
    top = torch.cat([lin[n_boundary:2*n_boundary], torch.full((n_boundary, 1), b, device=device)], dim=1)
    left = torch.cat([torch.full((n_boundary, 1), a, device=device), lin[2*n_boundary:3*n_boundary]], dim=1)
    right = torch.cat([torch.full((n_boundary, 1), b, device=device), lin[3*n_boundary:]], dim=1)
    # ... and connect them
    x_boundary = torch.cat([bottom, top, left, right], dim=0)

    t_boundary = torch.rand(n_boundary*4,1,device=device) * (t1-t0) + t0


    return x_interior,t_interior, x_boundary, t_boundary,  x_initial , t_initial
