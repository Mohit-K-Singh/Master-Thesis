import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri


def plot_slice_z0(domain, uh, N, error_L2, error_H1, z_slice=0.0, tol=1e-6):
    """Plot 2D slice of 3D solution at z=z_slice."""
    # Extract all mesh vertices
    coords = domain.geometry.x  # shape: (num_vertices, 3)
    
    # Find vertices near z=0 plane
    vertex_mask = np.abs(coords[:, 2] - z_slice) < tol
    slice_vertices = np.where(vertex_mask)[0]
    
    if len(slice_vertices) == 0:
        raise ValueError(f"No vertices found near z={z_slice}. Adjust tol or slice position.")
    
    # Get 2D coordinates of the slice
    x = coords[vertex_mask, 0]
    y = coords[vertex_mask, 1]
    
    # Create mapping from full mesh to slice vertices
    vertex_map = {v: i for i, v in enumerate(slice_vertices)}
    
    # Extract triangles contained in the slice
    domain.topology.create_connectivity(3, 0)  # Tet-to-vertex
    tets = domain.topology.connectivity(3, 0).array.reshape(-1, 4)
    
    slice_triangles = []
    for tet in tets:
        # Find vertices of this tet that are in the slice
        in_slice = [v for v in tet if v in vertex_map]
        
        # A triangle exists if exactly 3 vertices are in the slice
        if len(in_slice) == 3:
            slice_triangles.append([vertex_map[v] for v in in_slice])
    
    if not slice_triangles:
        raise ValueError("No complete triangles found in slice. Try finer mesh.")
    
    triangles = np.array(slice_triangles)
    
    # Get solution values at slice vertices
    z_values = uh.x.array[slice_vertices]
    
    # Create triangulation and plot
    triang = tri.Triangulation(x, y, triangles)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(triang, z_values, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_title(f"Solution slice at z={z_slice} | {N} elements | L2 error: {error_L2:.5f} | H1 error: {error_H1:.5f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("$\psi(x,y,0)$")
    plt.show()
