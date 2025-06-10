import numpy as np
import torch
def compute_l2_error(model, grid_points, n_points, psi_true, volume_element):
    with torch.no_grad():
        psi = model(grid_points).detach().reshape(n_points, n_points, n_points).cpu().numpy()
    sign = np.sign(np.sum(psi * psi_true))
    psi = sign * psi

    # Calculate volume element for normalization and l2 error

    #Normalization
    psi = psi/np.sqrt(np.sum(psi**2) * volume_element)

    # L2 error
    l2_error = np.sqrt(np.sum((psi - psi_true)**2) * volume_element)

    return l2_error

def compute_l2_h1_error(model, grid_points, n_points, psi_true, grad_psi_true, volume_element):
    grid_points = grid_points.requires_grad_(True)

    # Compute model prediction and its gradient
    psi_pred = model(grid_points)
    grads = torch.autograd.grad(
        outputs=psi_pred,
        inputs=grid_points,
        grad_outputs=torch.ones_like(psi_pred),
        create_graph=False,
        retain_graph=False,
        only_inputs=True
    )[0]

    # Reshape and move to CPU for NumPy
    psi_pred = psi_pred.detach().reshape(n_points, n_points, n_points).cpu().numpy()
    grads = grads.detach().reshape(n_points, n_points, n_points, 3).cpu().numpy()

    # Normalize predicted wavefunction
    sign = np.sign(np.sum(psi_pred * psi_true))
    psi_pred = sign * psi_pred
    psi_pred /= np.sqrt(np.sum(psi_pred**2) * volume_element)

    # L2 error
    l2_error = np.sum((psi_pred - psi_true)**2) * volume_element

    # H1 semi-norm (gradient difference)
    grad_diff = grads - grad_psi_true
    grad_error_sq = np.sum(np.sum(grad_diff**2, axis=-1)) * volume_element

    h1_error = np.sqrt(l2_error**2 + grad_error_sq)

    return l2_error, h1_error