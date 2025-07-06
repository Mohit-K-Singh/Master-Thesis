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
