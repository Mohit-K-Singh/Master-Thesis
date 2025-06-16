from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import assemble_matrix
from petsc4py.PETSc import ScalarType
from one_dim_plot import plot_wave

"""
This script computes the solution to the 1D-time independent Schrödinger equation. This is an eigenvalue problem and we use FeniCSx with SLEPc.
"""

# Set paramerts: Domain Length, number of finite elements and some constants
L = 5 # Domain 
N = 100  # No. of finite elements
mass = 1.0  # Particle mass
hbar = 1.0  # Reduced Planck constant

# Create mesh
domain = mesh.create_interval(MPI.COMM_WORLD, N, [-L, L])

# Define function space
V = functionspace(domain, ("Lagrange", 1))

# Create homogenous Dirichlet Boundary conditions.
facets = mesh.locate_entities_boundary(
    domain,
    dim=(domain.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], -L) | np.isclose(x[0], L),
)
dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=facets)
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# Define the potential: Harmonic oscillator
x = ufl.SpatialCoordinate(domain)
V_potential = 0.5 * x[0]**2  # V(x) = 0.5*x²

# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define the bilinear form (Hamiltonian)
a = 1/2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(V_potential * u, v) * ufl.dx

# Assemble the stiffness matrix; apply boundary conditions
A = assemble_matrix(fem.form(a), bcs = [bc])
A.assemble()

# Create the mass matrix for the eigenvalue problem; apply boundary conditions
m = ufl.inner(u, v) * ufl.dx
M = assemble_matrix(fem.form(m), bcs=[bc])
M.assemble()

# Create SLEPc eigenvalue solver; generates 5 eigenpairs; order s.t small eigenvalues are listed first
eps = SLEPc.EPS().create(domain.comm)
eps.setOperators(A, M)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps.setDimensions(5, PETSc.DECIDE)  # Find 5 eigenvalues
eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
eps.solve()

# Get number of converged eigenvalues
nconv = eps.getConverged()
print(f"Number of converged eigenvalues: {nconv}")


# Extract eigenvalues 
eigenvalues = []
for i in range(nconv):
    # Get eigenvalue
    eigval = eps.getEigenvalue(i)
    eigenvalues.append(eigval.real)
    
# Print eigenvalues, Sanity check : Should be multiples of 0.5
print("Computed eigenvalues:")
for i, val in enumerate(eigenvalues):
    print(f"E_{i}: {val:.4f}")

# I have come across soltuions, wich have smallest eigenvalues but do not have norm 1. This happens especially for d>1. This loop is a sanity check and filters out numerical artifacts.
for i in range(nconv):
    # Get first eigenfunction
    uh = fem.Function(V)
    vr, vi = A.createVecs()
    eps.getEigenpair(i, vr, vi)
 
    uh.x.array[:] = vr.array_r
    
    # Compute the norm and check if it is 1, wich is a condition on the wavefunction.
    norm = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx)))
    is_close_to_1 = np.isclose(norm, 1.0, atol= 1e-6, rtol=1e-6)
    if is_close_to_1:
        uh.x.array[:] /= norm
        break

# Define the analytical soltuion
V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: (1 / np.pi**0.25) * np.exp(-0.5 * x[0]**2))

# Compute the correct sign.
inner_product_form = fem.form(ufl.inner(uh, uex) * ufl.dx)
I = fem.assemble_scalar(inner_product_form)

correct_sign = np.sign(I)

if correct_sign <0 :
    uh.x.array[:] *= -1

# Now we can compute the error
L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

print("L2 error: ",error_L2)


H1_error = fem.form(ufl.inner(ufl.grad(uh-uex), ufl.grad(uh-uex)) * ufl.dx + ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local_H1 = fem.assemble_scalar(H1_error)
error_H1 = np.sqrt(domain.comm.allreduce(error_local_H1, op=MPI.SUM))

print("H1 error: ", error_H1)


# Plot the solution
plot_wave(V, V2, error_L2, error_H1, uh, uex, N)

    
    



