import numpy as np
import torch
from main import NeuralNetwork
import matplotlib.pyplot as plt

import numpy as np

def true_psi0_2d(xx, yy):
    """
    This is the true solution in 2D
    """
    return (1 / np.sqrt(np.pi)) * np.exp(-0.5 * (xx**2 + yy**2))


# Initialize the network and load our saved model
model = NeuralNetwork()
checkpoint = torch.load("best_model.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Specify our domain and create a meshgrid
L = 5  # Domain was [-5,5]²
n_points = 200
x = torch.linspace(-L, L, n_points)
y = torch.linspace(-L, L, n_points)
xx, yy = torch.meshgrid(x, y)
grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

# Determine our network function
with torch.no_grad():
    psi = model(grid_points).reshape(n_points, n_points).cpu().numpy()


# Get true solution
psi_true = true_psi0_2d(xx, yy).cpu().numpy()


# Determine the correct sign.
sign = np.sign(np.sum(psi * psi_true))
psi = sign * psi

# ======================================================= L2 and norm computation ========================================================
# Determine area element for computation of L2 error
dx = 2 * L / (n_points - 1)
dy = dx
area_element = dy*dx
psi = psi/np.sqrt(np.sum(psi**2) * area_element)
l2_error = np.sqrt(np.sum((psi - psi_true)**2) * area_element)
print(f"L2 error: {l2_error}")

# Check if norm is 1
norm_true = np.trapz(np.trapz(np.abs(psi_true)**2, dx=dy), dx=dx)
print(f"∫∫ |ψ_true|² dx dy = {norm_true:.6f} (should be 1.0)")

norm_pred = np.trapz(np.trapz(np.abs(psi)**2, dx=dy), dx=dx)
print(f"∫∫ |ψ_pred|² dx dy = {norm_pred:.6f} (should be 1.0)")
# ===================================================== Plotting ========================================================

# ===== First heatmaps ===============
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

im0 = axs[0].imshow(psi_true, extent=[-L, L, -L, L], origin='lower', cmap='BuPu')
axs[0].set_title("True Wavefunction")
axs[0].set_xlabel('x'); axs[0].set_ylabel('y')
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(psi, extent=[-L, L, -L, L], origin='lower', cmap='BuPu')
axs[1].set_title("Predicted Wavefunction")
axs[1].set_xlabel('x'); axs[1].set_ylabel('y')
fig.colorbar(im1, ax=axs[1])

plt.suptitle(f"2D Time independent Schrödinger equation | L2 Error: {l2_error}")
plt.tight_layout()
plt.show()


#============= Second contour plot ==============
density_pred = np.abs(psi)**2
density_true = np.abs(psi_true)**2
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

im0 = axs[0].contourf(density_true, extent=[-L, L, -L, L], origin='lower', cmap='BuPu')
axs[0].set_title("True Wavefunction")
axs[0].set_xlabel('x'); axs[0].set_ylabel('y')
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].contourf(density_pred, extent=[-L, L, -L, L], origin='lower', cmap='BuPu')
axs[1].set_title("Predicted Wavefunction")
axs[1].set_xlabel('x'); axs[1].set_ylabel('y')
fig.colorbar(im1, ax=axs[1])

plt.suptitle(fr"2D Time independent Schrödinger equation | Density: $|\psi(x)|²$")
plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(16, 6))

#=============== Third Surface plots =======d
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(xx.numpy(), yy.numpy(), psi, cmap='bone', rstride=1, cstride=1, alpha=0.9)
ax1.set_title("Predicted Wavefunction")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel(r'$\psi(x, y)$')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# True
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(xx.numpy(), yy.numpy(), psi_true, cmap='bone', rstride=1, cstride=1, alpha=0.9)
ax2.set_title("True Wavefunction")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel(r'$\psi_{true}(x, y)$')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
