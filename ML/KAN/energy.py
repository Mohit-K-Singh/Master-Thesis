import torch
import numpy as np
import matplotlib.pyplot as plt

def phase_evolution_loss1(model, x,t, omega=1.0):
    """
    Penalize deviation from expected global phase evolution, integrated over x and t.
    Makes sure we learn the right phase!
    2DTD Schrödinger equation with Harmonic oscilator as potential

    """
    X, Y ,T = torch.meshgrid(x, x, t, indexing='ij')
    xy_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    t_input = T.flatten().unsqueeze(-1)
    psi0 = model(xy_input, torch.zeros_like(t_input)).reshape(X.shape)

    psi_t = model(xy_input, t_input).reshape(X.shape)

    expected_phase = (-1 * omega * t_input).reshape(X.shape)
    relative_phase = torch.angle(psi_t * torch.conj(psi0))

    phase_error = (relative_phase - expected_phase)**2
    loss = torch.trapezoid(torch.trapezoid(torch.trapezoid(phase_error, x=x, dim=0),x=x, dim=0), x=t, dim=0)
    return loss


def phase_evolution_loss(model, xy,t, omega=1.0):
    """
    Penalize deviation from expected global phase evolution, integrated over x and t.
    Makes sure we learn the right phase!
    2DTD Schrödinger equation with Harmonic oscilator as potential

    """
    #x = x.unsqueeze(-1)
    #t = t.unsqueeze(-1)
    #xy_input = torch.stack([x,x], dim=1).squeeze()
    psi0 = model(xy, torch.zeros_like(t))

    psi_t = model(xy, t)

    expected_phase = (-1 * omega * t.squeeze())
    relative_phase = torch.angle(psi_t * torch.conj(psi0))

    phase_error = (relative_phase - expected_phase)**2
    return phase_error.mean()
    #loss = torch.trapezoid(torch.trapezoid(torch.trapezoid(phase_error, x=x, dim=0),x=x, dim=0), x=t, dim=0)
    return loss

def boundary_loss(model, xy, time):
    t,_ = torch.sort(time, dim=0)
    psi = model(xy, t)
    #loss = torch.trapezoid(torch.abs(psi)**2, x=t.squeeze(), dim=0)
    loss = (torch.abs(psi)**2).mean() 
    return loss
def initial_loss1(model, x, time, psi_initial):
    #x,_ = torch.sort(xy[:,0], dim=0)
    #y,_ = torch.sort(xy[:,1], dim=0)
    #t,_ = torch.sort(time, dim=0)

    #X, Y, T = torch.meshgrid(x, y, t.squeeze(), indexing='ij')
    X, Y, T = torch.meshgrid(x, x, time, indexing='ij')
    xy_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    t_input = T.flatten().unsqueeze(-1)
    psi = model(xy_input, t_input).reshape(X.shape)[:,:,0]
    psi_initial = psi_initial.reshape(psi.shape)
    
    loss = torch.trapezoid(torch.trapezoid(torch.abs(psi-psi_initial)**2, x=x, dim=0), x=x, dim=0)
    return loss
def initial_loss(model, xy, time, psi_initial):
    #x,_ = torch.sort(xy[:,0], dim=0) 
    #print(xy.shape, time.shape)
    psi = model(xy, time)
    #loss = torch.trapezoid(torch.abs(psi-psi_initial)**2, x=x, dim=0)
    #print(psi.shape, psi_initial.shape)
    loss = (torch.abs(psi-psi_initial)**2).mean() 
    return loss
def variational_loss(model, xy,time):
    x,_ = torch.sort(xy[:,0], dim=0) #shape 40
    y,_ = torch.sort(xy[:,1], dim=0)
    t,_ = torch.sort(time, dim=0)

    X, Y, T = torch.meshgrid(x, y, t.squeeze(), indexing='ij')

    xy_input = torch.stack([X.flatten(), Y.flatten()], dim=-1) #[64000,2]
    #print(xy_input.shape) 
    t_input = T.flatten().unsqueeze(-1)

    xy_input.requires_grad_(True)
    t_input.requires_grad_(True)

    psi = model(xy_input, t_input) # [64000]
    #print(psi.shape)

    V = 0.5 * (X**2 + Y**2)
    #print("V",V.shape)
    dpsi_real_dt = torch.autograd.grad(psi.real, t_input, grad_outputs=torch.ones_like(psi.real), create_graph=True)[0]
    dpsi_imag_dt = torch.autograd.grad(psi.imag, t_input, grad_outputs=torch.ones_like(psi.imag), create_graph=True)[0]

    dpsi_dt = torch.complex(dpsi_real_dt, dpsi_imag_dt)
    #print(dpsi_dt.shape)

    dpsi_real_dx = torch.autograd.grad(psi.real, xy_input, grad_outputs=torch.ones_like(psi.real), create_graph=True)[0][:,0]
    dpsi_imag_dx = torch.autograd.grad(psi.imag, xy_input, grad_outputs=torch.ones_like(psi.imag), create_graph=True)[0][:,0]

    dpsi_dx = torch.complex(dpsi_real_dx, dpsi_imag_dx) #[64000]
    #print(dpsi_dx.shape)


    dpsi_real_dy = torch.autograd.grad(psi.real, xy_input, grad_outputs=torch.ones_like(psi.real), create_graph=True)[0][:,1]
    dpsi_imag_dy = torch.autograd.grad(psi.imag, xy_input, grad_outputs=torch.ones_like(psi.imag), create_graph=True)[0][:,1]

    dpsi_dy = torch.complex(dpsi_real_dy, dpsi_imag_dy)

    psi = psi.reshape(X.shape) #[40,40,40]
    dpsi_dt = dpsi_dt.reshape(X.shape)
    dpsi_dx = dpsi_dx.reshape(X.shape)
    dpsi_dy = dpsi_dy.reshape(X.shape)

    expectation_integrand = 0.5*(torch.abs(dpsi_dx)**2 + torch.abs(dpsi_dy)**2) + V*torch.abs(psi)**2
    #print("exp", expectation_integrand.shape)
    Expectation = torch.trapezoid(torch.trapezoid(expectation_integrand,x=x, dim=0), x=y, dim=0)
    #print(Expectation.shape)
    inner = torch.trapezoid(torch.trapezoid( psi * torch.conj(dpsi_dt), x=x, dim=0), x=y, dim=0)
    inner2 = torch.trapezoid(torch.trapezoid( dpsi_dt * torch.conj(psi) , x=x, dim=0), x=y, dim=0) 
    #print(inner.shape)
    diff_inner = (1j/2 * (inner -inner2)).real
    #print(diff_inner)
    Langrangian = diff_inner - Expectation #shape 40
    #S = Langrangian.mean()
    #print(Langrangian.shape)
    S= torch.trapezoid(Langrangian,x=t.squeeze())
    #S = torch.mean(Langrangian)
    #grads = torch.autograd.grad(S, model.parameters(), create_graph=True)
    #print(type(grads), grads[1].shape)
    #print(grads.shape)
    #flat_grads = torch.cat([g.reshape(-1) for g in grads])
    #print(flat_grads.shape, flat_grads[1], flat_grads)
    #loss = torch.sum(torch.norm(flat_grads)**2)
    #return loss
    #print(loss)
    #print(S)
    return S


def tdse_residual_loss(model, xy,time):

    xy_input = xy
    t_input = time
    
    xy_input.requires_grad_(True)
    t_input.requires_grad_(True) 
    
    psi = model(xy_input, t_input) # [64000]
    #print(psi.shape)

    #V = 0.5 * (X**2 + Y**2)
    x = xy[:, 0]
    y = xy[:, 1]
    V = 0.5 * (x**2 + y**2)
    dpsi_real_dt = torch.autograd.grad(psi.real, t_input, grad_outputs=torch.ones_like(psi.real), create_graph=True)[0]
    dpsi_imag_dt = torch.autograd.grad(psi.imag, t_input, grad_outputs=torch.ones_like(psi.imag), create_graph=True)[0]

    dpsi_dt = torch.complex(dpsi_real_dt, dpsi_imag_dt)
    #print(dpsi_dt.shape)

    dpsi_real_dx = torch.autograd.grad(psi.real, xy_input, grad_outputs=torch.ones_like(psi.real), create_graph=True)[0][:,0]
    dpsi_imag_dx = torch.autograd.grad(psi.imag, xy_input, grad_outputs=torch.ones_like(psi.imag), create_graph=True)[0][:,0]

    dpsi_dx = torch.complex(dpsi_real_dx, dpsi_imag_dx) #[64000]
    #print(dpsi_dx.shape)


    dpsi_real_dy = torch.autograd.grad(psi.real, xy_input, grad_outputs=torch.ones_like(psi.real), create_graph=True)[0][:,1]
    dpsi_imag_dy = torch.autograd.grad(psi.imag, xy_input, grad_outputs=torch.ones_like(psi.imag), create_graph=True)[0][:,1]

    dpsi_dy = torch.complex(dpsi_real_dy, dpsi_imag_dy)


    # Second derivatives
    d2psi_real_dx2 = torch.autograd.grad(dpsi_real_dx, xy_input, grad_outputs=torch.ones_like(dpsi_real_dx), create_graph=True)[0][:,0]
    d2psi_imag_dx2 = torch.autograd.grad(dpsi_imag_dx, xy_input, grad_outputs=torch.ones_like(dpsi_imag_dx),create_graph=True)[0][:,0]

                                         
    d2psi_real_dy2 = torch.autograd.grad(dpsi_real_dy, xy_input, torch.ones_like(dpsi_real_dy), create_graph=True)[0][:, 1]
    d2psi_imag_dy2 = torch.autograd.grad(dpsi_imag_dy, xy_input, torch.ones_like(dpsi_imag_dy), create_graph=True)[0][:, 1]
    d2psi_dy2 = torch.complex(d2psi_real_dy2, d2psi_imag_dy2)

    d2psi_dx2 = torch.complex(d2psi_real_dx2, d2psi_imag_dx2)
    
    H_psi = -0.5 * (d2psi_dx2 + d2psi_dy2)+ V * psi
    
    # Residual attempt
    residual = 1j * dpsi_dt.squeeze() - H_psi
    loss = torch.mean(torch.abs(residual)**2)
    return loss
