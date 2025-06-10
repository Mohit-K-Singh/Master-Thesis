import torch

############## 1D- Time independent Schrödinger equation ##########################
def energy_loss(model, points):
    """
    V: Potential is derived from the harmonic oscillator
    H: Hamiltonian --> H = -h²/2m d²/dx² + V(x)
    This function calculates the energy expectation value: E = <psi,H|psi> / <psi,psi>, where <*,*> is the L² inner product.
    """
    points.requires_grad_(True)
    psi = model(points)
    V = 0.5 * points**2

    grad_psi = torch.autograd.grad(
            outputs=psi, inputs = points,
            grad_outputs = torch.ones_like(psi),
            create_graph = True, retain_graph = True
    )[0]

    # Avoid second order gradient via integration by parts. In the complex case one would takes torch.abs(grad_psi)**2
    kinetic = 0.5 * grad_psi**2
    potential = V * psi**2
    total_density = kinetic + potential
    total_energy = total_density.mean()
    norm = (psi**2).mean()
    loss = total_energy/norm

    return loss

def boundary_loss(model, boundary_points):
    """
    Calculates the loss on the boundary --> Is used for the boundary penalty method to enforce boundary conditions (Dirichlet for now)
    """
    psi_b = model(boundary_points)
    return torch.mean(psi_b**2)
