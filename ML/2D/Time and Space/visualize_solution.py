import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.integrate import trapezoid


from main import ComplexNetwork

# Set backend to avoid Qt issues (Linux)
os.environ['QT_QPA_PLATFORM'] = 'xcb'

def psi_true_2d(x, y, t, m=1.0, omega=1.0, hbar=1.0):
    """2D ground state wavefunction with time evolution."""
    prefactor = (m * omega / (np.pi * hbar))**0.5
    exp_space = torch.exp(-m * omega * (x**2 + y**2) / (2 * hbar))
    exp_time = torch.exp(-1j * omega * t)
    return prefactor * exp_space * exp_time

# Setup grid and time
n_points = 50
x = torch.linspace(-4, 4, n_points)
y = torch.linspace(-4, 4, n_points)
time_points = torch.linspace(0, 1, 50) 
X, Y , T= torch.meshgrid(x,y,time_points )#, indexing="ijl")

xy_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
t_input = T.flatten().unsqueeze(-1)
 # 30 frames

model = ComplexNetwork()
checkpoint = torch.load("best_model.pth", weights_only=True)
#checkpoint = torch.load("2_percent.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

model.eval()
with torch.no_grad():
    psi = model(xy_input, t_input).reshape(X.shape)
    psi_exact = psi_true_2d(X.flatten(), Y.flatten(), T.flatten()).reshape(X.shape)
    numerator1 = torch.trapezoid(torch.trapezoid(torch.abs(psi_exact - psi)**2, x=x, dim=0), x=y, dim=0)
    total = torch.sqrt(torch.trapezoid(numerator1, x=time_points))

psi_abs = torch.abs(psi)**2
# Optionally reconstruct complex psi normalized
integrals = []

# Loop over each time index and integrate over x
for i in range(T.shape[1]):  # time steps along axis 1
    prob_density = psi_abs[..., i]   # |psi|^2 at time t_i
    #integral = np.sum(prob_density) * dx
    inte = trapezoid(trapezoid(prob_density, x=x, axis=0), x=y)
    #print(integral, inte)
    integrals.append(inte)
plt.figure(figsize=(10, 6))
plt.plot(time_points.numpy(), integrals)
#plt.axhline(1.0, color='red', linestyle='--', label='Expected = 1')
plt.xlabel("Time t")
plt.ylabel(r"$\int |\psi(x,t)|^2 dx$")
plt.title("Normalization Check over Time")
plt.legend()
plt.grid(True)
plt.show()

        
# Initialize figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0, 0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('$\psi$')
""" 
# Pre-compute all frames
frames = []
for t in time_points:
    psi = psi_true_2d(X.flatten(), Y.flatten(), t).reshape(X.shape)
    #prob_density = torch.abs(psi)**2
    prob_density = psi.real
    # Create surface plot for this frame
    surf = ax.plot_surface(
        X.numpy(), Y.numpy(), prob_density.numpy(),
        cmap='viridis', edgecolor='none', alpha=0.9
    )
    title = ax.text(0.5, 1.05, 0.5, f'Time = {t:.2f}', 
                   transform=ax.transAxes, ha='center')
    frames.append([surf, title])  # Store artists for this frame

# Create animation
ani = ArtistAnimation(
    fig, 
    frames,
    interval=100,  # ms between frames
    blit=False,    # Required for 3D plots
    repeat=True
)

# Save and show
ani.save('wavefunction_evolution.gif', writer='pillow', fps=30, dpi=100)
plt.tight_layout()
plt.show() """

# Pre-compute all frames
frames = []
for i, t in enumerate(time_points):
    # Extract model prediction for this time step
    #prob_density = torch.abs(psi[..., i])**2  # Probability density
    prob_density = psi.real[..., i]  # Alternatively: plot real part
    # Create surface plot
    surf = ax.plot_surface(
        X[..., 0].numpy(),  # x grid (same for all times)
        Y[..., 0].numpy(),  # y grid (same for all times)
        prob_density.numpy(),
        cmap='viridis', 
        edgecolor='none', 
        alpha=0.9
    )
    
    # Add time annotation
    title = ax.text2D(
        0.5, 1.05, 
        f'Model Prediction | Time = {t:.2f} | L2 = {total:.4f}', 
        transform=ax.transAxes, 
        ha='center'
    )
    frames.append([surf, title])

# Create animation
ani = ArtistAnimation(
    fig, 
    frames,
    interval=100,  # ms between frames
    blit=False,    # Required for 3D plots
    repeat=True
)

# Save and show
ani.save('wave_evolution.gif', writer='pillow', fps=30, dpi=100)
plt.tight_layout()
plt.show()