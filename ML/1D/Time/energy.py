import torch
import numpy as np
from scipy.integrate import trapezoid

############## 1D- Time dependent Schrödinger equation ##########################

def boundary_loss(model, x,t):
    """
    Calculates the loss on the boundary --> Is used for the boundary penalty method to enforce boundary conditions (Dirichlet for now)
    """
    psi_b = model(x,t)
    return torch.mean(torch.abs(psi_b)**2)

def initial_loss(model, x,t, psi_initial):
    psi_0 = model(x,t)
    #x= initial_points[:,0:1]
    loss = torch.mean(torch.abs(psi_0 - psi_initial)**2)
    return loss


def psi0_fn(x):
    """
    Gaussian wave packet: ψ₀(x) = A exp(-x² / (2σ²)) * exp(i k₀ x)
    """
    sigma = 1.0
    k0 = 5.0
    A = (1.0 / (sigma * torch.sqrt(torch.tensor(np.pi))))**0.5
    real = A * torch.exp(-x**2 / (2 * sigma**2)) * torch.cos(k0 * x)
    imag = A * torch.exp(-x**2 / (2 * sigma**2)) * torch.sin(k0 * x)
    return torch.complex(real, imag)



def tdse_residual_loss(model, x,t):
    N_x = x.numel()
    x_max = torch.max(x).cpu().detach().numpy()
    x_min = torch.min(x).cpu().detach().numpy()
    t_max = torch.max(t).cpu().detach().numpy()
    t_min = torch.min(t).cpu().detach().numpy()
    dx = (x_max -x_min) /(N_x -1)
    dt = (t_max - t_min) /(N_x -1)
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    psi = model(x, t)
    #print("psi : ", psi.shape)
    psi_real =psi.real
    psi_imag =psi.imag
    phi = psi.detach().clone()  # test function (frozen)
    # Potential
    V = 0.5 * x**2
    #print("V: ", V.squeeze().shape)
    # Time derivative ∂ψ/∂t

    dpsi_real_dt = torch.autograd.grad(psi_real, t, grad_outputs=torch.ones_like(psi_real),
                                       create_graph=True)[0]#[:, 1:2]
    dpsi_imag_dt = torch.autograd.grad(psi_imag, t, grad_outputs=torch.ones_like(psi_imag),
                                       create_graph=True)[0]#[:, 1:2]
    dpsi_dt = torch.complex(dpsi_real_dt, dpsi_imag_dt)

    #print("Time derivative: ", dpsi_dt.shape)


    dpsi_real_dx = torch.autograd.grad(psi_real, x, grad_outputs=torch.ones_like(psi_real),
                                       create_graph=True)[0]#[:, 0:1]
    dpsi_imag_dx = torch.autograd.grad(psi_imag, x, grad_outputs=torch.ones_like(psi_imag),
                                       create_graph=True)[0]#[:, 0:1]

    # Second derivatives
    d2psi_real_dx2 = torch.autograd.grad(dpsi_real_dx, x, grad_outputs=torch.ones_like(dpsi_real_dx),
                                         create_graph=True)[0]#[:, 0:1]
    d2psi_imag_dx2 = torch.autograd.grad(dpsi_imag_dx, x, grad_outputs=torch.ones_like(dpsi_imag_dx),
                                         create_graph=True)[0]#[:, 0:1]

    
    # Recombine into complex second derivative
    d2psi_dx2 = torch.complex(d2psi_real_dx2, d2psi_imag_dx2)
    #print("Second space : ", d2psi_dx2.squeeze().shape)
    # Hamiltonian
    H_psi = -0.5 * d2psi_dx2.squeeze() + V.squeeze() * psi.squeeze()
    #print("H_psi: ", H_psi.shape)
    
    # Residual attempt
    residual = 1j * dpsi_dt.squeeze() - H_psi
    #loss = torch.mean(torch.abs(residual)**2)
    loss = torch.trapezoid(torch.abs(residual)**2)
    
    return loss


def weak_loss(model, x, t):
    N_x = x.numel()
    x_max = torch.max(x).cpu().detach().numpy()
    x_min = torch.min(x).cpu().detach().numpy()
    t_max = torch.max(t).cpu().detach().numpy()
    t_min = torch.min(t).cpu().detach().numpy()
    dx = (x_max -x_min) /(N_x -1)
    dt = (t_max - t_min) /(N_x -1)
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    psi = model(x, t)
    #print("psi : ", psi.shape)
    psi_real =psi.real
    psi_imag =psi.imag
    phi = psi.detach().clone()  # test function (frozen)
    # Potential
    V = 0.5 * x**2
    #print("V: ", V.squeeze().shape)
    # Time derivative ∂ψ/∂t

    dpsi_real_dt = torch.autograd.grad(psi_real, t, grad_outputs=torch.ones_like(psi_real),
                                       create_graph=True)[0]#[:, 1:2]
    dpsi_imag_dt = torch.autograd.grad(psi_imag, t, grad_outputs=torch.ones_like(psi_imag),
                                       create_graph=True)[0]#[:, 1:2]
    #dpsi_dt = dpsi_real_dt + 1j * dpsi_imag_dt
    dpsi_dt = torch.complex(dpsi_real_dt, dpsi_imag_dt)

    #print("Time derivative: ", dpsi_dt.shape)
    #dpsi_dt = torch.autograd.grad(psi, t, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]

    # Second spatial derivative ∂²ψ/∂x²
    #dpsi_dx = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi),retain_graph=True, create_graph=True)[0]
    #d2psi_dx2 = torch.autograd.grad(dpsi_dx, x, grad_outputs=torch.ones_like(psi),retain_graph=True, create_graph=True)[0]

    dpsi_real_dx = torch.autograd.grad(psi_real, x, grad_outputs=torch.ones_like(psi_real),
                                       create_graph=True)[0]#[:, 0:1]
    dpsi_imag_dx = torch.autograd.grad(psi_imag, x, grad_outputs=torch.ones_like(psi_imag),
                                       create_graph=True)[0]#[:, 0:1]

    # Second derivatives
    d2psi_real_dx2 = torch.autograd.grad(dpsi_real_dx, x, grad_outputs=torch.ones_like(dpsi_real_dx),
                                         create_graph=True)[0]#[:, 0:1]
    d2psi_imag_dx2 = torch.autograd.grad(dpsi_imag_dx, x, grad_outputs=torch.ones_like(dpsi_imag_dx),
                                         create_graph=True)[0]#[:, 0:1]

    
    # Recombine into complex second derivative
    #d2psi_dx2 = d2psi_real_dx2 + 1j * d2psi_imag_dx2
    d2psi_dx2 = torch.complex(d2psi_real_dx2, d2psi_imag_dx2)
    #print("Second space : ", d2psi_dx2.squeeze().shape)
    # Hamiltonian
    H_psi = -0.5 * d2psi_dx2.squeeze() + V.squeeze() * psi.squeeze()
    #print("H_psi: ", H_psi.shape)
    
    
    # Weak integrand
    #print(torch.conj(phi).shape)
    #integrand = 1j * dpsi_dt.squeeze() * torch.conj(phi) - H_psi * torch.conj(phi)
    
    #print(integrand.shape)
    #integral = torch.sum(integrand) *   1/len(x)
    #integral = integrand.mean()
    #integral = torch.trapezoid(integrand)
    #integral = torch.sum(torch.abs(integrand)) * dx * dt
    
    
    # Residual attempt
    residual = 1j * dpsi_dt.squeeze() - H_psi
    #loss = torch.mean(torch.abs(residual)**2)
    loss = torch.trapezoid(torch.abs(residual)**2)
    
    return loss
    #return torch.abs(integral)  # squared modulus
    #return integral
