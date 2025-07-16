import numpy as np
import torch
def true_psi0(xx, yy, zz):
    """
    Returns the 3d groundstate soltuion
    """
    return (1 / np.pi**(3/4)) * torch.exp(-0.5 * (xx**2 + yy**2 + zz**2))

def compute_l2_error(model, device):
    x = torch.linspace(-3.5,3.5,50).to(device)
    X,Y,Z = torch.meshgrid(x,x,x, indexing='ij')
    xyz = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
    psi_true = true_psi0(X,Y,Z)
    with torch.no_grad():
        psi = model(xyz).reshape(X.shape)
        sign = torch.sign(torch.sum(psi*psi_true))
        psi = sign*psi

        diff = (psi - psi_true)
        l2_error = torch.sqrt(torch.trapezoid(torch.trapezoid(torch.trapezoid(torch.abs(diff)**2,x=x,dim=0),x=x,dim=0),x=x, dim=0))
        return l2_error


def conservation_loss(model,device):
    x = torch.linspace(-3.5,3.5,50).to(device)
    X,Y,Z = torch.meshgrid(x,x,x, indexing='ij')
    xyz = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
    psi = model(xyz).reshape(X.shape)
    norm = torch.sqrt(torch.trapezoid(torch.trapezoid(torch.trapezoid(torch.abs(psi)**2,x=x,dim=0),x=x,dim=0),x=x, dim=0))
    norm_loss = (norm-1)**2
    return norm_loss