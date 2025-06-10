import matplotlib.tri as tri
import numpy as np
from dolfinx import fem
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_wave(V, V2, error_L2, error_H1, uh, uex, N):
    x_coords = V.tabulate_dof_coordinates()[:, 0]
    sorted_indices = np.argsort(x_coords)
    uex_interpolated = fem.Function(V)
    uex_interpolated.interpolate(lambda x: (1 / np.pi**0.25) * np.exp(-0.5 * x[0]**2))

    x2_coords = V2.tabulate_dof_coordinates()[:, 0]
    sorted2_indices = np.argsort(x2_coords)
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords[sorted_indices], uh.x.array[sorted_indices], 
            'b-', linewidth=2, label="Computed")
    # Plot exact solution (using same sorted coordinates)
    plt.plot(x2_coords[sorted2_indices], uex.x.array[sorted2_indices],
            'r--', linewidth=2, label="Exact solution")
    plt.title(f"1D Solution | {N} elements | L2 error: {error_L2:.5f} | H1 error: {error_H1:.5f}")
    plt.xlabel("x")
    plt.ylabel("$\psi(x)$")
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.grid(True)
    plt.legend()
    plt.show()
