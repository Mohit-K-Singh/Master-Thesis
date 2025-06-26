import matplotlib.tri as tri
import numpy as np
from dolfinx import fem
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_surface(domain, uh, error_L2, error_H1, N):
    # Extract mesh geometry
    coords = domain.geometry.x  
    # Extract cell-node connectivity for triangles
    # cell_dim = 2 (triangles), entity_dim = 0 (vertices)
    domain.topology.create_connectivity(2, 0)
    cells = domain.topology.connectivity(2, 0).array.reshape(-1, 3)

    # Extract values of uh at DoFs (Lagrange P1 → vertex-based)
    z = uh.x.array
    # Extract x, y for triangulation
    x = coords[:, 0]
    y = coords[:, 1]

    triang = tri.Triangulation(x, y, cells)

    # Plot
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(triang, z, cmap="viridis", linewidth=0.2)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_title(f"First Eigenfunction | {N} elements | L2 error: {error_L2:.5f} | H1 error: {error_H1:.5f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("ψ(x, y)")
    plt.tight_layout()
    plt.show()

def compare(domain, uh, V):
    # Extract mesh data
    coords = domain.geometry.x
    domain.topology.create_connectivity(2, 0)
    cells = domain.topology.connectivity(2, 0).array.reshape(-1, 3)
    x, y = coords[:, 0], coords[:, 1]
    triang = tri.Triangulation(x, y, cells)

    # Get solutions
    z_uh = uh.x.array  # FEM solution
    uex = fem.Function(V)
    uex.interpolate(lambda x: (1 / np.sqrt(np.pi)) * np.exp(-0.5 * (x[0]**2 + x[1]**2)))
    z_uex = uex.x.array  # Exact solution interpolated to FEM space

    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 6))

    # Plot FEM solution
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_trisurf(triang, z_uh, cmap=cm.viridis, linewidth=0.2)
    ax1.set_title("Computed Solution (FEM)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("ψ(x,y)")
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # Plot exact solution
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_trisurf(triang, z_uex, cmap=cm.viridis, linewidth=0.2)
    ax2.set_title("Exact Solution (Analytical)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("ψ(x,y)")
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    plt.tight_layout()
    plt.show()