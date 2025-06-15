
import torch
import matplotlib.pyplot as plt
import numpy as np
from main import ComplexNetwork
from matplotlib.animation import FuncAnimation
from scipy.integrate import trapezoid

def psi0_fn(x):
    """
    Gaussian wave packet: ψ₀(x) = A exp(-x² / (2σ²)) * exp(i k₀ x)
    """
    sigma = 1.0
    k0 = 5.0
    A = (1.0 / (sigma * torch.sqrt(torch.tensor(np.pi))))**0.5
    real = A * torch.exp(-x**2 / (2 * sigma**2)) * torch.cos(k0 * x)
    imag = A * torch.exp(-x**2 / (2 * sigma**2)) * torch.sin(k0 * x)
    return torch.complex(real, imag)
def psi_true(x, t, m=1.0, omega=1.0, hbar=1.0):
    """
    Ground state wavefunction of 1D quantum harmonic oscillator (complex-valued)
    """
    prefactor = (m * omega / (np.pi * hbar))**0.25
    exp_space = torch.exp(-m * omega * x**2 / (2 * hbar))
    exp_time = torch.exp(-1j * omega * t / 2)
    return prefactor * exp_space * exp_time



# Create a meshgrid of x and t values
x = torch.linspace(-7, 7, 200)
t = torch.linspace(0, 7, 100)
X, T = torch.meshgrid(x, t, indexing='ij')

# Flatten the grid for model input
x_flat = X.reshape(-1, 1)
t_flat = T.reshape(-1, 1)
xt_input = torch.cat([x_flat, t_flat], dim=1)

# Evaluate model (without gradient)
model = ComplexNetwork()
checkpoint = torch.load("best_model.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

model.eval()
with torch.no_grad():
    psi = model(x_flat, t_flat)
    psi_init = psi0_fn(x_flat)
    psi_exact = psi_true(x_flat, t_flat).reshape(X.shape)
    psi_real = psi.real.reshape(X.shape).cpu().numpy()
    psi_imag = psi.imag.reshape(X.shape).cpu().numpy()
    psi_abs = psi_real**2 + psi_imag**2  # amplitude


dx = (x[1] - x[0]).item()
#norm_constants = np.sum(psi_abs) * dx 
norm_constants = trapezoid(psi_abs, x.numpy(), axis=0) # numpy array
psi_real = psi_real / np.sqrt(norm_constants)[None, :]
psi_imag = psi_imag / np.sqrt(norm_constants)[None, :]
#print(trapezoid(psi_real**2 + psi_imag**2, x=x, axis=0))
psi = torch.complex(torch.from_numpy(psi_real), torch.from_numpy(psi_imag))
psi_abs = psi_real**2 + psi_imag**2 

#print(trapezoid(torch.abs(psi)**2, x=x, axis=0))

#These two also same
numerator1 = trapezoid(torch.abs(psi_exact - psi)**2, x=x, axis=0)
numerator = trapezoid((psi_exact.real - psi.real)**2 + (psi_exact.imag - psi.imag)**2, x=x, axis=0)
total = trapezoid(numerator, x=t)


# These two the same
denominator2 = trapezoid(torch.abs(psi_exact)**2, x=x, axis=0)
denominator3 = trapezoid(psi_exact.real**2 + psi_exact.imag**2, x = x, axis=0)
l2_error = np.sqrt(numerator) # / denominator2)
print(total)
#for i in range(T.shape[1]):
    


plt.figure(figsize=(10, 6))
plt.plot(t.numpy(), l2_error)
#plt.axhline(1.0, color='red', linestyle='--', label='Expected = 1')
plt.xlabel("Time t")
plt.ylabel("L2")
plt.title("L2 Check over Time")
plt.legend()
plt.grid(True)
plt.savefig("L2_check.png")
plt.show()




psi_exact_abs = psi_exact.real**2 + psi_exact.imag**2
    
# Optionally reconstruct complex psi normalized
integrals = []

# Loop over each time index and integrate over x
for i in range(T.shape[1]):  # time steps along axis 1
    prob_density = psi_abs[:, i]   # |psi|^2 at time t_i
    #integral = np.sum(prob_density) * dx
    inte = trapezoid(prob_density, x=x)
    #print(integral, inte)
    integrals.append(inte)
plt.figure(figsize=(10, 6))
plt.plot(t.numpy(), integrals)
#plt.axhline(1.0, color='red', linestyle='--', label='Expected = 1')
plt.xlabel("Time t")
plt.ylabel(r"$\int |\psi(x,t)|^2 dx$")
plt.title("Normalization Check over Time")
plt.legend()
plt.grid(True)
plt.show()


fig, axes = plt.subplots(3, 2, figsize=(14, 12))
(ax1, ax1b), (ax2, ax2b), (ax3, ax3b) = axes

## Want to plo these in another column
psi2_real = psi_exact.real
psi2_imag = psi_exact.imag
psi2_abs = psi_exact_abs

# Initial plots
line_real, = ax1.plot(x, psi_real[:, 0], 'b')
line_imag, = ax2.plot(x, psi_imag[:, 0], 'r')
line_abs, = ax3.plot(x, psi_abs[:, 0], 'k')

line_real2, = ax1b.plot(x, psi2_real[:, 0], 'b')
line_imag2, = ax2b.plot(x, psi2_imag[:, 0], 'r')
line_abs2,  = ax3b.plot(x, psi2_abs[:, 0], 'k')

# Set titles and labels
ax1.set_title("Predicted")
ax1.set_ylabel("Re(ψ)")
ax1.set_ylim(psi_real.min(), psi_real.max())

#ax2.set_title("Imaginary part of ψ(x,t)")
ax2.set_ylabel("Im(ψ)")
ax2.set_ylim(psi_imag.min(), psi_imag.max())

#ax3.set_title("Absolute value |ψ(x,t)|")
ax3.set_ylabel("|ψ|")
ax3.set_xlabel("Position x")
ax3.set_ylim(psi_abs.min(), psi_abs.max())


# Titles and labels for second column
ax1b.set_title("Truth")
ax1b.set_ylim(psi2_real.min(), psi2_real.max())

#ax2b.set_title("Imaginary part of ψ₂(x,t)")
ax2b.set_ylim(psi2_imag.min(), psi2_imag.max())

#ax3b.set_title("Absolute value |ψ₂(x,t)|")
ax3b.set_xlabel("Position x")
ax3b.set_ylim(psi2_abs.min(), psi2_abs.max())


# Time annotation
time_text = ax3.text(0.02, 0.95, '', transform=ax3.transAxes)

def init():
    line_real.set_ydata([np.nan] * len(x))
    line_imag.set_ydata([np.nan] * len(x))
    line_abs.set_ydata([np.nan] * len(x))
    line_real2.set_ydata([np.nan] * len(x))
    line_imag2.set_ydata([np.nan] * len(x))
    line_abs2.set_ydata([np.nan] * len(x))
    time_text.set_text('')
    return line_real, line_imag, line_abs, line_real2, line_imag2, line_abs2, time_text

def update(frame):
    line_real.set_ydata(psi_real[:, frame])
    line_imag.set_ydata(psi_imag[:, frame])
    line_abs.set_ydata(psi_abs[:, frame])
    line_real2.set_ydata(psi2_real[:, frame])
    line_imag2.set_ydata(psi2_imag[:, frame])
    line_abs2.set_ydata(psi2_abs[:, frame])
    time_text.set_text(f'Time t = {t[frame]:.2f}')
    return line_real, line_imag, line_abs, line_real2, line_imag2, line_abs2, time_text

# Create animation
ani = FuncAnimation(fig, update, frames=len(t), init_func=init,
                    blit=True, interval=50)

plt.tight_layout()
plt.show()
ani.save('solution_dual_column.mp4', writer='ffmpeg', fps=30)

