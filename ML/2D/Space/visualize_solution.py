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
L = 4  # Domain was [-5,5]²
n_points = 50
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
diff = np.abs(psi - psi_true)
l2_error = np.sqrt(np.trapezoid(np.trapezoid(diff**2, x=x),x=y))
print(f"L2 error: {l2_error}")


h = x[1] - x[0]
dpsi_true_x, dpsi_true_y = np.gradient(psi_true, h, edge_order=2)

# For predicted solution
dpsi_x, dpsi_y = np.gradient(psi, h, edge_order=2)

# Compute differences in gradients
diff_grad_x = np.abs(dpsi_x - dpsi_true_x)
diff_grad_y = np.abs(dpsi_y - dpsi_true_y)

# Compute H1 error (L2 norm of function + L2 norm of gradient)
h1_error = np.sqrt(
    np.trapezoid(np.trapezoid(diff**2, x=x), x=y) +  # L2 part
    np.trapezoid(np.trapezoid(diff_grad_x**2, x=x), x=y) +  # Gradient x part
    np.trapezoid(np.trapezoid(diff_grad_y**2, x=x), x=y)   # Gradient y part
)


# Check if norm is 1
norm_true = np.trapezoid(np.trapezoid(np.abs(psi_true)**2, x=x), x=y)
print(f"∫∫ |ψ_true|² dx dy = {norm_true:.6f} (should be 1.0)")

norm_pred = np.trapezoid(np.trapezoid(np.abs(psi)**2, x=x), x=y)
print(f"∫∫ |ψ_pred|² dx dy = {norm_pred:.6f} (should be 1.0)")
# ===================================================== Plotting ========================================================

# ===== First heatmaps ===============
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

im0 = axs[0].imshow(psi_true, extent=[-L, L, -L, L], origin='lower', cmap='BuPu')
axs[0].set_title(r" $|\psi_{true}(x)|²$")
axs[0].set_xlabel('x'); axs[0].set_ylabel('y')
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(psi, extent=[-L, L, -L, L], origin='lower', cmap='BuPu')
axs[1].set_title(r" $|\psi_{pred}(x)|²$")
axs[1].set_xlabel('x'); axs[1].set_ylabel('y')
fig.colorbar(im1, ax=axs[1])

plt.suptitle(f"L2 Error: {l2_error:.5f} | H1 Error: {h1_error:.4f}")
plt.tight_layout()
plt.savefig('heatmap.png')
plt.show()



#============= Second contour plot ==============
density_pred = np.abs(psi)**2
density_true = np.abs(psi_true)**2
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

im0 = axs[0].contourf(density_true, extent=[-L, L, -L, L], origin='lower', cmap='BuPu')
axs[0].set_title(r" $|\psi_{true}(x)|²$")
axs[0].set_xlabel('x'); axs[0].set_ylabel('y')
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].contourf(density_pred, extent=[-L, L, -L, L], origin='lower', cmap='BuPu')
axs[1].set_title(r" $|\psi_{pred}(x)|²$")
axs[1].set_xlabel('x'); axs[1].set_ylabel('y')
fig.colorbar(im1, ax=axs[1])

plt.suptitle(fr" Density: $|\psi(x)|²$")
plt.tight_layout()
plt.savefig('contour.png')
plt.show()



#=============== Third Surface plots =======d
fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(xx.numpy(), yy.numpy(), psi, cmap='magma', rstride=1, cstride=1, alpha=0.9)
ax1.set_title("Predicted Wavefunction")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel(r'$\psi_{pred}(x, y)$')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# True
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(xx.numpy(), yy.numpy(), psi_true, cmap='magma', rstride=1, cstride=1, alpha=0.9)
ax2.set_title("True Wavefunction")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel(r'$\psi_{true}(x, y)$')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig('surface.png')
plt.show()
