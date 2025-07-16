import torch


############## 2D- Time independent Schrödinger equation ##########################

def energy_loss(model, points):
    """
    Similiar to the 1D case we compute the expected Energy
    V: Potential is derived from the harmonic oscillator
    H: Hamiltonian --> H = -h²/2m d²/dx² + V(x)
    This function calculates the energy expectation value: E = <psi,H psi> / <psi,psi>, where <*,*> is the L² inner product.

    """
    # This method is simpler and just uses the fact that torch.mean() for random sampled points approximated the monte carlo integral
    # Multiplcation with the surface area can be left out( it would cancel out )
    points.requires_grad_(True)
    psi = model(points) # shape [3000,1]
    V = 0.5 *(points**2).sum(dim=1) #shape [3000]
    grad_psi = torch.autograd.grad(
            outputs=psi, inputs = points,
            grad_outputs = torch.ones_like(psi),
            create_graph = True, retain_graph = True
    )[0]  #shape [3000,2]


    # Avoid second order gradient via integration by parts. In the complex case one would takes torch.abs(grad_psi)**2
    kinetic = 0.5 * (grad_psi**2).sum(dim=1) # shape [3000]
    potential = V * (psi**2).squeeze() #shape[3000]
    total_density = kinetic + potential #shape[3000]
    total_energy = total_density.mean() 
    norm = (psi**2).mean() 
    loss = total_energy/norm
    #print(psi.shape, grad_psi.shape, potential.shape, total_density.shape, total_energy.shape)
    return loss
    """ 
    # This method spans a meshgrid and then integrates over the x and y dimensions. Gives sub 1% L2 error.
    x,_ = torch.sort(points[:,0], dim=0) 
    y,_ = torch.sort(points[:,1], dim=0)

    X,Y = torch.meshgrid(x,y, indexing='ij')
    xy_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    xy_input.requires_grad_(True)
    psi = model(xy_input)
    dpsi_dx = torch.autograd.grad(psi, xy_input, grad_outputs=torch.ones_like(psi), create_graph=True)[0][:,0]
    dpsi_dy = torch.autograd.grad(psi, xy_input, grad_outputs=torch.ones_like(psi), create_graph=True)[0][:,1]

    V = 0.5 * (X**2 + Y**2)
    psi = psi.reshape(X.shape) #[40,40,40]
    dpsi_dy = dpsi_dy.reshape(X.shape)
    dpsi_dx = dpsi_dx.reshape(X.shape)

    #print(V.shape, psi.shape, dpsi_dx.shape, dpsi_dy.shape)

    Expectation_integrand= 0.5 * (dpsi_dx**2 + dpsi_dy**2) + V*psi**2
    Expectation = torch.trapezoid(torch.trapezoid(Expectation_integrand,x=x, dim=0), x=y, dim=0)
    norm = torch.trapezoid(torch.trapezoid(torch.abs(psi)**2, x=x, dim=0), x=y, dim=0)


    return Expectation/norm """


def boundary_loss(model, boundary_points):
    """
    Boundary term for the boundary penalt method to enforce Dirichlet conditions
    """
    psi_b = model(boundary_points)
    return torch.mean(psi_b**2)
