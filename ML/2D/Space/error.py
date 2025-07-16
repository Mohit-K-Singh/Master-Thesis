import numpy as np
import torch



def true_psi0(xx, yy):
    """
    This is the true solution in 2D
    """
    return (1 / np.sqrt(np.pi)) * torch.exp(-0.5 * (xx**2 + yy**2))
def convservation_loss(model, device):
    x = torch.linspace(-4,4,20).to(device)
    X,Y = torch.meshgrid(x,x, indexing='ij')
    xy =torch.stack([X.flatten(), Y.flatten()], dim=-1)
    psi = model(xy).reshape(X.shape)
    norm = torch.sqrt(torch.trapezoid(torch.trapezoid(torch.abs(psi)**2, x=x, dim=0), x=x, dim=0))
    norm_loss = (norm-1)**2
    return norm_loss

def compute_error(model,device):
    x = torch.linspace(-4,4,40).to(device)
    X,Y = torch.meshgrid(x,x, indexing='ij')
    xy =torch.stack([X.flatten(), Y.flatten()], dim=-1)
    psi_true = true_psi0(X,Y)
    with torch.no_grad():
        psi = model(xy).reshape(X.shape)
        sign = torch.sign(torch.sum(psi * psi_true))
        psi = sign * psi
        diff = (psi - psi_true)
        l2_error = torch.sqrt(torch.trapezoid(torch.trapezoid(torch.abs(diff)**2, x=x, dim=0),x=x, dim=0))
        return l2_error