import numpy as np
import torch
from main import NeuralNetwork
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def true_psi0(x):
    """
    Computes the true solution
    """
    return (1 / np.pi**0.25) * np.exp(-0.5 * x**2)

# Initialize model
model = NeuralNetwork()
checkpoint = torch.load("best_model.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

a, b = -5.0, 5.0
num_points = 500

x_plot = torch.linspace(a, b, num_points).view(-1,1)

# Get neural network function
with torch.no_grad():
    psi = model(x_plot).squeeze().cpu().numpy()

x_np = x_plot.squeeze().cpu().numpy()

psi_true = true_psi0(x_np)
# Normalize function ---> Is this cheating?
psi = psi / np.sqrt(np.trapz(psi**2, x_np))
# Determine the correct sign...
sign = np.sign(np.sum(psi * psi_true))
# ... then compute L2 error
psi = sign * psi
diff = (psi - psi_true)**2
l2_error = np.sqrt(np.trapz(diff, x_np))
# Computation of Relative L2 error
l2_norm_psi_true = np.sqrt(np.trapz(np.abs(psi_true)**2, x_np))
relative_l2_error = l2_error / l2_norm_psi_true
# Compute H1 error
h = x_np[1] - x_np[0]
dpsi_true = np.gradient(psi_true, h, edge_order=2)
dpsi = np.gradient(psi, h, edge_order=2)
h1_error = np.sqrt(np.trapz((psi-psi_true)**2, x_np) + np.trapz((dpsi - dpsi_true)**2, x_np))
# Check if norm is 1
norm_pred = np.trapz(np.abs(psi)**2, x_np)
norm_true = np.trapz(np.abs(psi_true)**2, x_np)
print(f"∫∫ |ψ_true|² dx dy = {norm_true:.6f} (should be 1.0)")
print(f"∫∫ |ψ_pred|² dx dy = {norm_pred:.6f} (should be 1.0)")


# Print the errors
print(f"L2 error: {l2_error}\nRelative L2 error: {relative_l2_error} \nH1 error: {h1_error}")
# Plot function and density
plt.figure(figsize=(10, 5))
plt.plot(x_np, psi, label=r'$\psi(x)$ (learned)', linestyle = 'dashdot')
plt.plot(x_np, psi**2, label=r'$|\psi(x)|^2$ (learned)', linestyle='dashdot')
plt.plot(x_np, psi_true, label=r'$\psi_0(x)$ (Analytic)', linestyle='--')
plt.plot(x_np, psi_true**2, label=r'$|\psi(x)|^2$ (analytic)', linestyle='--')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.title(rf'1D-time independent Schrödinger equation | L2-Error ={l2_error:.5f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
