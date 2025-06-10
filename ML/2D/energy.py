import torch


############## 2D- Time independent Schrödinger equation ##########################

def energy_loss(model, points):
    """
    Similiar to the 1D case we compute the expected Energy
    V: Potential is derived from the harmonic oscillator
    H: Hamiltonian --> H = -h²/2m d²/dx² + V(x)
    This function calculates the energy expectation value: E = <psi,H psi> / <psi,psi>, where <*,*> is the L² inner product.

    """
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

    return loss


def boundary_loss(model, boundary_points):
    """
    Boundary term for the boundary penalt method to enforce Dirichlet conditions
    """
    psi_b = model(boundary_points)
    return torch.mean(psi_b**2)
