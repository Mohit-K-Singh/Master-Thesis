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



def compute_error(model, device):
    x = torch.linspace(-6, 6, 100).to(device)
    t = torch.linspace(0, 1, 100).to(device)
    X, T = torch.meshgrid(x, t, indexing='ij')

    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)
    with torch.no_grad():
        psi = model(x_flat, t_flat).reshape(X.shape)
        psi_exact = psi_true(x_flat, t_flat).reshape(X.shape)
        numerator1 = torch.trapezoid(torch.abs(psi_exact - psi)**2, x=x, axis=0)
        total = torch.sqrt(torch.trapezoid(numerator1, x=t))

    return total