import numpy as np
import torch


def psi_true(x, y, t, m=1.0, omega=1.0, hbar=1.0):
    """
    Ground state wavefunction of 2D quantum harmonic oscillator (complex-valued)
    """
    prefactor = (m * omega / (np.pi * hbar))**0.5
    exp_space = torch.exp(-m * omega * (x**2 +y**2)/ (2 * hbar))
    exp_time = torch.exp(-1j * omega * t)
    return prefactor * exp_space * exp_time    

def conservation_loss(model,device,x_range=(-3.5, 3.5), num_points=30):
    x = torch.linspace(*x_range, num_points).to(device)
    y = torch.linspace(*x_range, num_points).to(device)
    t = torch.linspace(0, 1, num_points).to(device)
    X, Y, T = torch.meshgrid(x,y, t, indexing="ij")
    x_flat, y_flat, t_flat = X.flatten().unsqueeze(-1), Y.flatten().unsqueeze(-1), T.flatten().unsqueeze(-1)
    xy_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    psi = model(xy_input, t_flat)
    psi = psi.reshape(X.shape)
    prob_density = torch.abs(psi)**2
    norm = torch.trapezoid(torch.trapezoid(prob_density, x=x, dim = 0), x=y, dim=0) 
    final_norm = torch.sqrt(torch.trapezoid(norm, x=t))
    norm_loss = (final_norm - 1.0)**2
    return norm_loss

def compute_error(model, device, num_points =40):
    x = torch.linspace(-3.5, 3.5, num_points).to(device)
    t = torch.linspace(0, 1, num_points).to(device)
    X, Y, T = torch.meshgrid(x, x, t.squeeze())
    xy_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    t_input = T.flatten().unsqueeze(-1)
    with torch.no_grad():
        psi = model(xy_input, t_input).reshape(X.shape)
        psi_exact = psi_true(X.flatten(), Y.flatten(), T.flatten()).reshape(X.shape)
        numerator1 = torch.trapezoid(torch.trapezoid(torch.abs(psi_exact - psi)**2, x=x, dim=0), x=x, dim=0)
        total = torch.sqrt(torch.trapezoid(numerator1, x=t))

    return total