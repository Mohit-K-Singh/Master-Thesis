import torch
import numpy as np
import matplotlib.pyplot as plt

def phase_evolution_loss(model, x,t, omega=1.0):
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


def phase_evolution_loss1(model, x,t, omega=1.0):
    """
    Penalize deviation from expected global phase evolution, integrated over x and t.
    Makes sure we learn the right phase!
    2DTD Schrödinger equation with Harmonic oscilator as potential

    """
    x = x.unsqueeze(-1)
    t = t.unsqueeze(-1)
    xy_input = torch.stack([x,x], dim=1).squeeze()
    psi0 = model(xy_input, torch.zeros_like(t))

    psi_t = model(xy_input, t)

    expected_phase = (-1 * omega * t.squeeze())
    relative_phase = torch.angle(psi_t * torch.conj(psi0))

    phase_error = (relative_phase - expected_phase)**2
    return phase_error.mean()
    #loss = torch.trapezoid(torch.trapezoid(torch.trapezoid(phase_error, x=x, dim=0),x=x, dim=0), x=t, dim=0)
    return loss
def boundary_loss2(model, xy, time):
    x,_ = torch.sort(xy[:,0], dim=0)
    y,_ = torch.sort(xy[:,1], dim=0)
    t,_ = torch.sort(time, dim=0)
    # Create full meshgrid
    X, Y, T = torch.meshgrid(x, y, t.squeeze(), indexing='ij')
    
    # Identify boundary points (where x=a, x=b, y=a, or y=b)
    a, b = x.min().item(), x.max().item()
    is_boundary = ((X == a) | (X == b) | (Y == a) | (Y == b))
    
    # Extract boundary points
    xy_boundary = torch.stack([X[is_boundary], Y[is_boundary]], dim=-1)
    t_boundary = T[is_boundary].unsqueeze(-1)
    # Evaluate model on boundary
    psi_boundary = model(xy_boundary, t_boundary)
    psi_boundary = psi_boundary.reshape(len(t), -1)  # Group by time
    loss = torch.abs(psi_boundary)**2
    return loss.mean() 


def boundary_loss1(model, xy, time):
    x,_ = torch.sort(xy[:,0], dim=0)
    y,_ = torch.sort(xy[:,1], dim=0)
    t,_ = torch.sort(time, dim=0)

    X, Y, T = torch.meshgrid(x, y, t.squeeze(), indexing='ij')
    xy_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    t_input = T.flatten().unsqueeze(-1)
    psi = model(xy_input, t_input) #.reshape(X.shape)
    """ # Convert to numpy for plotting
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    # Flatten the grid (optional, but useful if you want to see all points)
    x_flat = X_np.flatten()
    y_flat = Y_np.flatten()

    # Create a 2D scatter plot of the grid points
    plt.figure(figsize=(8, 6))
    plt.scatter(x_flat, y_flat, s=5, c='b', alpha=0.6, label='Grid Points')

    # Mark boundary points (optional, if you want to distinguish them)
    a, b = x.min().item(), x.max().item()  # Assuming [a, b] is your spatial domain
    boundary_mask = (
        (x_flat == a) | (x_flat == b) | 
        (y_flat == a) | (y_flat == b)
    )
    plt.scatter(
        x_flat[boundary_mask], 
        y_flat[boundary_mask], 
        s=20, c='r', alpha=1.0, label='Boundary Points'
    )

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Grid Points (Sanity Check)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show() """
    #loss_space= torch.trapezoid(torch.trapezoid(torch.abs(psi)**2, x=t.squeeze(), dim=2), x=y, dim=1)
    #loss = torch.trapezoid(loss_space, x=y, dim=0)
    loss = torch.trapezoid(torch.abs(psi)**2, x=X.flatten(), dim=0)
    return loss
def boundary_loss(model, xy, time):
    t,_ = torch.sort(time, dim=0)
    psi = model(xy, t)
    #loss = torch.trapezoid(torch.abs(psi)**2, x=t.squeeze(), dim=0)
    loss = (torch.abs(psi)**2).mean()
    return loss
def initial_loss(model, x, time, psi_initial):
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
def variational_loss(model, xy,time):
    x,_ = torch.sort(xy[:,0], dim=0)
    y,_ = torch.sort(xy[:,1], dim=0)
    t,_ = torch.sort(time, dim=0)

    X, Y, T = torch.meshgrid(x, y, t.squeeze(), indexing='ij')

    xy_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    #print(xy_input.shape)
    t_input = T.flatten().unsqueeze(-1)

    xy_input.requires_grad_(True)
    t_input.requires_grad_(True)

    psi = model(xy_input, t_input)
    #print(psi.shape)

    V = 0.5 * (X**2 + Y**2)
    #print("V",V.shape)
    dpsi_real_dt = torch.autograd.grad(psi.real, t_input, grad_outputs=torch.ones_like(psi.real), create_graph=True)[0]
    dpsi_imag_dt = torch.autograd.grad(psi.imag, t_input, grad_outputs=torch.ones_like(psi.imag), create_graph=True)[0]

    dpsi_dt = torch.complex(dpsi_real_dt, dpsi_imag_dt)
    #print(dpsi_dt.shape)

    dpsi_real_dx = torch.autograd.grad(psi.real, xy_input, grad_outputs=torch.ones_like(psi.real), create_graph=True)[0][:,0]
    dpsi_imag_dx = torch.autograd.grad(psi.imag, xy_input, grad_outputs=torch.ones_like(psi.imag), create_graph=True)[0][:,0]

    dpsi_dx = torch.complex(dpsi_real_dx, dpsi_imag_dx)
    #print(dpsi_dx.shape)


    dpsi_real_dy = torch.autograd.grad(psi.real, xy_input, grad_outputs=torch.ones_like(psi.real), create_graph=True)[0][:,1]
    dpsi_imag_dy = torch.autograd.grad(psi.imag, xy_input, grad_outputs=torch.ones_like(psi.imag), create_graph=True)[0][:,1]

    dpsi_dy = torch.complex(dpsi_real_dy, dpsi_imag_dy)

    psi = psi.reshape(X.shape)
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
    Langrangian = diff_inner - Expectation
    S= torch.trapezoid(Langrangian,x=t.squeeze())
    #print(S)
    return S
