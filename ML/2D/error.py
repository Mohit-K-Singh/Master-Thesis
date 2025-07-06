import numpy as np
import torch


def psi_true(x, y, t, m=1.0, omega=1.0, hbar=1.0):
    """
    Ground state wavefunction of 1D quantum harmonic oscillator (complex-valued)
    """
    prefactor = (m * omega / (np.pi * hbar))**0.5
    exp_space = torch.exp(-m * omega * (x**2 +y**2)/ (2 * hbar))
    exp_time = torch.exp(-1j * omega * t)
    return prefactor * exp_space * exp_time    



def compute_error(model, device):
    x = torch.linspace(-4, 4, 25).to(device)
    t = torch.linspace(0, 1, 25).to(device)
    X, Y, T = torch.meshgrid(x, x, t.squeeze())
    xy_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    t_input = T.flatten().unsqueeze(-1)
    with torch.no_grad():
        psi = model(xy_input, t_input).reshape(X.shape)
        psi_exact = psi_true(X.flatten(), Y.flatten(), T.flatten()).reshape(X.shape)
        numerator1 = torch.trapezoid(torch.trapezoid(torch.abs(psi_exact - psi)**2, x=x, dim=0), x=x, dim=0)
        total = torch.sqrt(torch.trapezoid(numerator1, x=t))

    return total