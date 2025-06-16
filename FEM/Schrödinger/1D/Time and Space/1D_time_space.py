from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import assemble_matrix
from petsc4py.PETSc import ScalarType
print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == 'c'

from one_dim_plot import plot_solution, plot_solution_with_exact, plot_solution_together

"""
This script aims to solve the 1D- time dependent Schrödinger equation.

"""


# Step size and total time
dt = 0.01
T = 4*np.pi
num_steps = int(T/dt)

L = 5 # Domain 
N = 100
domain = mesh.create_interval(MPI.COMM_WORLD, N, [-L, L])
V = functionspace(domain, ("Lagrange", 1))

psi = fem.Function(V)
psi.interpolate(lambda x: (1/np.pi)**0.25 * np.exp(-0.5 * x[0]**2))  # Ground state of harmonic oscillator


# Create homogenous Dirichlet Boundary conditions.
facets = mesh.locate_entities_boundary(
    domain,
    dim=(domain.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], -L) | np.isclose(x[0], L),
)
dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=facets)
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

x = ufl.SpatialCoordinate(domain)
V_potential = 0.5 * x[0]**2 

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = 1/2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(V_potential * u, v) * ufl.dx

# Assemble the stiffness matrix; apply boundary conditions
H = assemble_matrix(fem.form(a), bcs = [bc])
H.assemble()

I_form = fem.form(ufl.inner(u, v) * ufl.dx)
I = assemble_matrix(I_form, bcs=[bc])
I.assemble()

LHS_mat = I.copy()
RHS_mat = I.copy()
H_mult = H.copy()
H_mult.scale(1j * dt/2)

LHS_mat.axpy(1.0, H_mult, structure=PETSc.Mat.Structure.SAME)
RHS_mat.axpy(-1.0, H_mult, structure=PETSc.Mat.Structure.SAME)

#Solver 
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(LHS_mat)
solver.setType(PETSc.KSP.Type.GMRES)
solver.getPC().setType(PETSc.PC.Type.NONE)

psi_new = fem.Function(V)
rhs_vec = RHS_mat.createVecRight()

wavefunctions = []
wavefunctions2 = []
#x_coords = V.tabulate_dof_coordinates()
x_coords = V.tabulate_dof_coordinates()[:, 0]
#print(x_coords.shape)
sorted_indices = np.argsort(x_coords)
x_coords = x_coords[sorted_indices]

V2 = fem.functionspace(domain, ("Lagrange", 2)) 
x2_coords = V2.tabulate_dof_coordinates()[:, 0]
sorted2_indices = np.argsort(x2_coords) 
x2_coords = x2_coords[sorted2_indices]
psi_exact = fem.Function(V2)
t = 0.0
total_error_l2, total_error_h1 = 0.0 , 0.0
print(num_steps)
for step in range(num_steps):
    RHS_mat.mult(psi.x.petsc_vec, rhs_vec)
    solver.solve(rhs_vec, psi_new.x.petsc_vec)
    psi_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    psi.x.array[:] = psi_new.x.array[:]
    psi_exact.interpolate(lambda x: (1 / np.pi**0.25) * np.exp(-0.5 * x[0]**2) * np.exp(-1j * 0.5 * t))

    if step % 5 == 0:
        wavefunctions.append(psi.x.array.copy())
        wavefunctions2.append(psi_exact.x.array.copy())

    if step % 50 == 0 and domain.comm.rank == 0:
        print(f"Step {step, num_steps}, t = {t:.2f}")


    t += dt

    L2_error = fem.form(ufl.inner(psi - psi_exact, psi - psi_exact) * ufl.dx)
    error_local = fem.assemble_scalar(L2_error)
    error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))


    H1_error = fem.form(ufl.inner(ufl.grad(psi-psi_exact), ufl.grad(psi-psi_exact)) * ufl.dx + ufl.inner(psi - psi_exact, psi - psi_exact) * ufl.dx)
    error_local_H1 = fem.assemble_scalar(H1_error)
    error_H1 = np.sqrt(domain.comm.allreduce(error_local_H1, op=MPI.SUM))
    
    total_error_l2 += np.real(error_L2)* dt 
    total_error_h1 += np.real(error_H1) * dt

final_l2 = np.sqrt(total_error_l2)
final_h1 = np.sqrt(total_error_h1)
print("L2 error: ", final_l2)
print("H1 error: ", final_h1)

#plot_solution(x_coords, wavefunctions, sorted_indices)
plot_solution_with_exact(x_coords, wavefunctions, sorted_indices, x2_coords,wavefunctions2 ,sorted2_indices, final_l2, final_h1)
#plot_solution_together(x_coords, wavefunctions, sorted_indices, x2_coords,wavefunctions2 ,sorted2_indices)
