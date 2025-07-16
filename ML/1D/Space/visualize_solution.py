import numpy as np
import torch
from main import NeuralNetwork
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def true_psi0(x):
    """
    Computes the true solution
    """
    return (1 / np.pi**0.25) * torch.exp(-0.5 * x**2)

# Initialize model
model = NeuralNetwork()
checkpoint = torch.load("bestes_modell.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
best_l2 =checkpoint['l2']   
print(best_l2)
print(checkpoint['epoch'], checkpoint['loss'])
model.eval()

a, b = -5.0, 5.0
num_points = 200

x = torch.linspace(-5,5,400)
# Get neural network function
with torch.no_grad():
    psi = model(x.unsqueeze(-1)).squeeze()
    psi_true = true_psi0(x)
    print(psi.shape, psi_true.shape)

    # Determine the correct sign...
    sign = torch.sign(torch.sum(psi * psi_true))
    # ... then compute L2 error
    psi = sign * psi
    diff = (psi - psi_true)
    l2_error = torch.sqrt(torch.trapezoid(torch.abs(diff)**2, x=x, dim=0))
    print(l2_error.item())

""" 
x = torch.linspace(-5,5,200)

with torch.no_grad():
    psi_true = true_psi0(x)
    psi = model(x.unsqueeze(-1)).squeeze()
    sign = torch.sign(torch.sum(psi * psi_true))
    psi = sign * psi
    diff = (psi -psi_true)
    l2_error = torch.sqrt(torch.trapezoid(torch.abs(diff)**2, x=x, dim=0))
    print(l2_error)
 """


psi_true_np = psi_true.cpu().numpy()
psi_np = psi.cpu().numpy()
# Compute H1 error
diff = np.abs(psi_np - psi_true_np)
h = x[1] - x[0]
dpsi_true = np.gradient(psi_true_np, h, edge_order=2)
dpsi = np.gradient(psi_np, h, edge_order=2)
diff_gradient = np.abs(dpsi - dpsi_true)

h1_error = np.sqrt(np.trapezoid(diff**2, x) + np.trapezoid(diff_gradient**2, x))
# Check if norm is 1
norm_pred = torch.trapezoid(torch.abs(psi)**2, x=x, dim=0)
norm_true = torch.trapezoid(torch.abs(psi_true)**2, x=x, dim=0)
print((torch.sqrt(norm_pred)-1)**2)
print(f"∫ |ψ_true|² dx  = {norm_true:.6f} (should be 1.0)")
print(f"∫ |ψ_pred|² dx  = {norm_pred:.6f} (should be 1.0)")


# Print the errors
print(f"L2 error: {l2_error}\nH1 error: {h1_error}")
# Plot function and density
plt.figure(figsize=(10, 5))
plt.plot(x, psi, label=r'$\psi_{pred}(x)$', linestyle = 'dashdot')
plt.plot(x, psi**2, label=r'$|\psi_{pred}(x)|^2$', linestyle='dashdot')
plt.plot(x, psi_true, label=r'$\psi(x)$ ', linestyle='dotted')
plt.plot(x, psi_true**2, label=r'$|\psi(x)|^2$ ', linestyle='dotted' )
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.title(rf'L2-Error ={l2_error:.5f} | H1-Error = {h1_error:.5f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('1d_space_comparison.png')
plt.show()
