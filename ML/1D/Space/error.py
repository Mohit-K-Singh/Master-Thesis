import numpy as np
import torch

def true_psi0(x):
    """
    Computes the true solution
    """
    return (1 / np.pi**0.25) * torch.exp(-0.5 * x**2)
def convservation_loss(model, device):
    x = torch.linspace(-5,5,200).to(device)
    psi = model(x.unsqueeze(-1)).squeeze()
    norm = torch.sqrt(torch.trapezoid(torch.abs(psi)**2, x=x, dim=0))
    norm_loss = (norm-1)**2
    return norm_loss

def compute_error(model,device):
    x = torch.linspace(-5,5,200).to(device)
    psi_true = true_psi0(x)
    with torch.no_grad():
        psi = model(x.unsqueeze(-1)).squeeze()
        sign = torch.sign(torch.sum(psi * psi_true))
        psi = sign * psi
        diff = (psi - psi_true)
        l2_error = torch.sqrt(torch.trapezoid(torch.abs(diff)**2, x=x, dim=0))
        return l2_error