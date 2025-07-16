import numpy as np
import torch


def psi_true(x, t, m=1.0, omega=1.0, hbar=1.0):
    """
    Ground state wavefunction of 1D quantum harmonic oscillator (complex-valued)
    """
    prefactor = (m * omega / (np.pi * hbar))**0.25
    exp_space = torch.exp(-m * omega * x**2 / (2 * hbar))
    exp_time = torch.exp(-1j * omega * t / 2)
    return prefactor * exp_space * exp_time


def conservation_loss(model,device,x_range=(-5, 5), num_points=50):
    x = torch.linspace(*x_range, num_points).to(device)
    t = torch.linspace(0, 1, num_points).to(device)
    X, T = torch.meshgrid(x, t, indexing='ij')
    #print(X.shape, T.shape)
    x_flat, t_flat = X.flatten().unsqueeze(-1), T.flatten().unsqueeze(-1)
    #print(x.shape)
    psi = model(x_flat, t_flat)
    psi = psi.reshape(X.shape)
    prob_density = torch.abs(psi)**2
    norm = torch.trapezoid(prob_density, x=x, dim = 0) 
    final_norm = torch.sqrt(torch.trapezoid(norm, x=t))
    norm_loss = (final_norm - 1.0)**2
    return norm_loss


def compute_error(model, device, num_points =100):
    x = torch.linspace(-5, 5, num_points).to(device)
    t = torch.linspace(0, 1, num_points).to(device)
    X, T = torch.meshgrid(x, t, indexing='ij')

    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)
    with torch.no_grad():
        psi = model(x_flat, t_flat).reshape(X.shape)
        psi_exact = psi_true(x_flat, t_flat).reshape(X.shape)
        numerator1 = torch.trapezoid(torch.abs(psi_exact - psi)**2, x=x, axis=0)
        total = torch.sqrt(torch.trapezoid(numerator1, x=t))

    return total