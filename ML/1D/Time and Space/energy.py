import torch
import numpy as np
from scipy.integrate import trapezoid

############## 1D- Time dependent Schrödinger equation ##########################

def boundary_loss(model, x,t):
    """
    Calculates the loss on the boundary --> Is used for the boundary penalty method to enforce boundary conditions (Dirichlet for now)
    """
    xd,td = x.squeeze(), t.squeeze()
    #print("xd:", xd.shape, "td:", td.shape)
    X, T = torch.meshgrid(xd, td, indexing = "ij")
    x, t = X.flatten().unsqueeze(-1), T.flatten().unsqueeze(-1)
    
    psi_boundary = model(x,t)
    psi_b = psi_boundary.reshape(X.shape)
    psi_space = torch.trapezoid(torch.abs(psi_b)**2, x=xd, dim=0)
    psi_time = torch.trapezoid(psi_space, x=td)
    #print("psi_time:", psi_time)
    return torch.abs(psi_time)
    #return torch.trapezoid(torch.abs(psi_b)**2)
    #return torch.mean(torch.abs(psi_b)**2)
def phase_evolution_loss1(model, x_ref, t_ref, omega=1.0):
    """
    Penalize deviation from correct global phase evolution.
    Expected: psi(x,t) = psi(x,0) * exp(-i omega t / 2)
    """

    with torch.no_grad():
        psi0 = model(x_ref, torch.zeros_like(t_ref))

    psi_t = model(x_ref, t_ref)
    
    # Expected phase shift
    expected_phase = -0.5 * omega * t_ref.squeeze()

    # Compute actual phase shift
    relative_phase = torch.angle(psi_t * torch.conj(psi0))
    # Penalize deviation
    return torch.trapezoid((relative_phase - expected_phase) ** 2, x=t_ref.squeeze())

def phase_evolution_loss(model, x_refs, t_refs, omega=1.0):
    """
    Penalize deviation from expected global phase evolution, integrated over x and t.
    Makes sure we learn the right phase!
    Inputs:
    - x_refs: [Nx, 1]
    - t_refs: [Nt, 1]
    """

    Nx, Nt = x_refs.size(0), t_refs.size(0)

    # 1. Create meshgrid of shape [Nx, Nt]
    X, T = torch.meshgrid(x_refs.squeeze(), t_refs.squeeze(), indexing='ij')
    x_grid = X.reshape(-1, 1)  # shape [Nx*Nt, 1]
    t_grid = T.reshape(-1, 1)  # shape [Nx*Nt, 1]

    with torch.no_grad():
        psi0 = model(x_grid, torch.zeros_like(t_grid))

    psi_t = model(x_grid, t_grid)

    # 2. Compute relative phase
    expected_phase = -0.5 * omega * t_refs.squeeze()  # shape [Nx*Nt]
    relative_phase = torch.angle(psi_t * torch.conj(psi0))  # shape [Nx*Nt]

    # 3. Reshape to [Nx, Nt]
    rel_phase_grid = relative_phase.view(Nx, Nt)
    expected_grid = expected_phase.view(1, Nt).expand(Nx, Nt)

    # 4. Integrate over x and t using trapezoidal rule
    phase_error = (rel_phase_grid - expected_grid)**2
    dx = x_refs[1] - x_refs[0]
    dt = t_refs[1] - t_refs[0]
    loss = torch.trapezoid(torch.trapezoid(phase_error, x=t_refs.squeeze(), dim=1), x=x_refs.squeeze(), dim=0)

    return loss
def initial_loss(model, x,t, psi_initial):

    psi_0 = model(x,t)

    #print("Initial_loss:",psi_0.shape, psi_initial.shape)
    #x= initial_points[:,0:1]
    loss = torch.trapezoid(torch.abs(psi_0 - psi_initial)**2, x=x.squeeze())
    #phase = torch.angle(psi_0 * torch.conj(psi_initial))
    #phase_loss = torch.trapezoid(torch.sin(phase)**2, x=x.squeeze())
    #phase_loss = torch.trapezoid(1 - torch.cos(phase), x=x.squeeze())
    #print(phase_loss, loss)
    #print("initial_loss:", loss.shape)
    return loss # + phase_loss



def variational_loss(model, x,t):
    xd,td = x.squeeze(), t.squeeze()
    #print("xd:", xd.shape, "td:", td.shape)
    X, T = torch.meshgrid(xd, td, indexing = "ij")
    #print(X.shape, T.shape)
    x, t = X.flatten().unsqueeze(-1), T.flatten().unsqueeze(-1)
    #print(x.shape)
    x.requires_grad_(True)
    t.requires_grad_(True)
    psi = model(x, t) 
    #print("psi : ", psi.shape)
    V = 0.5 * X**2 #shape [10000,1]
    #print("V: ", V.shape)
    dpsi_real_dt = torch.autograd.grad(psi.real, t, grad_outputs=torch.ones_like(psi.real),
                                       create_graph=True)[0]
    dpsi_imag_dt = torch.autograd.grad(psi.imag, t, grad_outputs=torch.ones_like(psi.imag),
                                       create_graph=True)[0]
    dpsi_dt = torch.complex(dpsi_real_dt, dpsi_imag_dt)
    #print("dpsi_dt: ", dpsi_dt.shape) #shape 

    dpsi_real_dx = torch.autograd.grad(psi.real, x, grad_outputs=torch.ones_like(psi.real),
                                       create_graph=True)[0]
    dpsi_imag_dx = torch.autograd.grad(psi.imag, x, grad_outputs=torch.ones_like(psi.imag),
                                       create_graph=True)[0]


    dpsi_dx = torch.complex(dpsi_real_dx, dpsi_imag_dx) #shape 
    #print("dpsi_dx: ", dpsi_dx.shape)

    psi = psi.reshape(X.shape)
    dpsi_dt = dpsi_dt.reshape(X.shape)
    dpsi_dx = dpsi_dx.reshape(X.shape)
    #print("psi : ", psi.shape)
    #print("dpsi_dt: ", dpsi_dt.shape) 
    #print("dpsi_dx: ", dpsi_dx.shape)
    #print("psi at 0 : ",psi[:,0].shape, psi[2,0])
    prob_density = torch.abs(psi)**2
    norm = torch.trapezoid(prob_density, x=xd, dim = 0) #scalar
    norm_loss = (norm - 1.0)**2
    #print("Density", prob_density.shape, prob_density.min(), prob_density.max())
    #print("Norm: ", norm.shape, norm[3], norm.min())

    expectation_integrand = 0.5*torch.abs(dpsi_dx)**2  + V * torch.abs(psi)**2 
    #print("Integrand:" , expectation_integrand.shape)
    Expectation = torch.trapezoid(expectation_integrand, x=xd, dim = 0) #/ norm #scalar
    #print("Expectation:", Expectation.shape)

    inner = torch.trapezoid( psi * torch.conj(dpsi_dt), x=xd, dim=0) #scalar
    inner2 = torch.trapezoid( dpsi_dt * torch.conj(psi) , x=xd, dim=0) #scalar
    #print("inner:", inner[0], inner.shape)
    diff_inner = (1j/2 * (inner - inner2)).real #/ norm #scalar
    
    Lagrangian = diff_inner - Expectation #scalar
    #Now i want to integrate the Lagrangian over Time

    #return torch.mean(Lagrangian**2)
    S = torch.trapezoid(Lagrangian,x=td)
    #print("Inner: ", inner.shape)
    #print("Inner2: ", inner2.shape)
    #print("diff: ", diff_inner.shape)
    #print("Langrangian: ", Lagrangian.shape)
    #print("S: ", S)
    return S
    return torch.abs(S)
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
