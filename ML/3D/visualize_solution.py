import numpy as np
import torch
from main import NeuralNetwork
import matplotlib.pyplot as plt



def true_psi0_3d(xx, yy, zz):
    """
    Returns the 3d groundstate soltuion
    """
    return (1 / np.pi**(3/4)) * np.exp(-0.5 * (xx**2 + yy**2 + zz**2))

# Initialize model and load our network
model = NeuralNetwork()
checkpoint = torch.load("best_model.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Specify the domain and create a meshgrid
L = 2.5  # Domain size [-L, L] x [-L, L]
n_points = 50
x = torch.linspace(-L, L, n_points)
y = torch.linspace(-L, L, n_points)
z = torch.linspace(-L, L, n_points)
xx, yy , zz= torch.meshgrid(x, y, z)
grid_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)  # Shape [n_points^2, 2]

# Get our network function
with torch.no_grad():
    psi = model(grid_points).reshape(n_points, n_points, n_points).cpu().numpy()

# Get the true ground state
psi_true = true_psi0_3d(xx, yy, zz).cpu().numpy()

# Determine the sign
sign = np.sign(np.sum(psi * psi_true))
psi = sign * psi


# Calculate volume element for normalization and l2 error
dx = 2 * L / (n_points - 1)
volume_element = dx **3

#Normalization
psi = psi/np.sqrt(np.sum(psi**2) * volume_element)

# L2 error
l2_error = np.sqrt(np.sum((psi - psi_true)**2) * volume_element)
print(f"L2 error: {l2_error}")

# Check if norm is 1
norm_true = np.trapezoid(np.trapezoid(np.trapezoid(np.abs(psi_true)**2, dx=dx), dx=dx) , dx=dx)
print(f"∫∫ |ψ_true|² dx dy = {norm_true:.6f} (should be 1.0)")
norm_pred = np.trapezoid(np.trapezoid(np.trapezoid(np.abs(psi)**2, dx=dx), dx=dx) , dx=dx)
print(f"∫∫ |ψ_pred|² dx dy = {norm_pred:.6f} (should be 1.0)")


# Take a slice and plot the resulting surface
z_index = 0
fig = plt.figure(figsize=(16, 6))

# ========= Predicted Solution ==============
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(xx[:, :, z_index].numpy(),
                         yy[:, :, z_index].numpy(),
                         psi[:, :, z_index],
                         cmap='viridis')
ax1.set_title(f"Predicted Wavefunction (z=0) | L2 error {l2_error}")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel(r"$\psi(x, y, 0)$")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# ============== True Solution =====================
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(xx[:, :, z_index].numpy(),
                         yy[:, :, z_index].numpy(),
                         psi_true[:, :, z_index],
                         cmap='viridis')
ax2.set_title("True Wavefunction (z=0)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel(r"$\psi_{\mathrm{true}}(x, y, 0)$")
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
