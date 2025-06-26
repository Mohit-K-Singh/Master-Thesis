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

"""
This script aims to solve the 2D- time dependent Schrödinger equation.

"""

# Step size and total time
dt = 0.001
T = 1
num_steps = int(T/dt)

#Domain size
L = 6
N = 100
domain = mesh.create_rectangle(MPI.COMM_WORLD, points = ((-L,-L), (L,L)), n = (N,N), cell_type = mesh.CellType.triangle)

V = functionspace(domain, ("Lagrange", 1))

# Create psi and apply initial conditions
psi = fem.Function(V)
psi.interpolate(lambda x: 1 / (np.sqrt(np.pi)) * np.exp(-0.5 * (x[0]**2 + x[1]**2)))

# Apply boundary conditions
facets = mesh.locate_entities_boundary(
        domain,
        dim=(domain.topology.dim -1),
        marker=lambda x: np.isclose(x[0], -L) | np.isclose(x[0], L) | np.isclose(x[1], -L) | np.isclose(x[1], L),
)
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V) 

x = ufl.SpatialCoordinate(domain)
V_potential = 0.5 * (x[0]**2 + x[1]**2)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

#Weak formulation
a = 1/2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(V_potential * u, v) * ufl.dx
#Assemble the stiffnes matrix
H = assemble_matrix(fem.form(a), bcs = [bc])
H.assemble()

I_form = fem.form(ufl.inner(u,v) * ufl.dx)
I = assemble_matrix(I_form, bcs=[bc])
I.assemble()

LHS_mat = I.copy()
RHS_mat = I.copy()
H_mult = H.copy()
H_mult.scale(1j* dt/2)

LHS_mat.axpy(1.0, H_mult, structure= PETSc.Mat.Structure.SAME)
RHS_mat.axpy(-1.0, H_mult, structure= PETSc.Mat.Structure.SAME)

#Initialize solver 
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(LHS_mat)
solver.setType(PETSc.KSP.Type.GMRES)
solver.getPC().setType(PETSc.PC.Type.NONE)

#Dummy variable
psi_new = fem.Function(V)
rhs_vec = RHS_mat.createVecRight()

#List to collect the solution
wavefunctions =[]
wavefunctions2 =[]

#Define coordinates and define exact solution
x_coords = V.tabulate_dof_coordinates()[:,0]
sorted_indices = np.argsort(x_coords)
x_coords = x_coords[sorted_indices]

V2 = fem.functionspace(domain, ("Lagrange", 2))
x2_coords = V2.tabulate_dof_coordinates()[:,0]
sorted2_indices = np.argsort(x2_coords)
x2_coords = x2_coords[sorted2_indices]
psi_exact = fem.Function(V2)

t= 0.0

l2_array = np.empty([num_steps+1])
h1_array = np.empty([num_steps+1])
for step in range(num_steps+1):
    RHS_mat.mult(psi.x.petsc_vec, rhs_vec)
    solver.solve(rhs_vec, psi_new.x.petsc_vec)
    psi_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode = PETSc.ScatterMode.FORWARD)
    psi.x.array[:] = psi_new.x.array[:]
    psi_exact.interpolate(lambda x: (1/ np.sqrt(np.pi)) * np.exp(-0.5 *(x[0]**2 + x[1]**2)) * np.exp(-1j * t))

    if step % 5 == 0:
        wavefunctions.append(psi.x.array.copy())
        wavefunctions2.append(psi_exact.x.array.copy())

    if step % 50 == 0 and domain.comm.rank == 0:
        print(f"Step {step, num_steps}, t = {t:.2f}")


    t += dt

    L2_error = fem.form(ufl.inner(psi - psi_exact, psi - psi_exact) * ufl.dx)
    error_local = fem.assemble_scalar(L2_error)
    error_L2 = domain.comm.allreduce(error_local, op=MPI.SUM)
    l2_array[step] = np.abs(error_L2)
    print(np.abs(error_L2), error_L2.real, error_L2)
    H1_error = fem.form(ufl.inner(ufl.grad(psi-psi_exact), ufl.grad(psi-psi_exact)) * ufl.dx + ufl.inner(psi - psi_exact, psi - psi_exact) * ufl.dx)
    error_local_H1 = fem.assemble_scalar(H1_error)
    error_H1 = domain.comm.allreduce(error_local_H1, op=MPI.SUM)
    h1_array[step] = np.abs(error_H1)
   

final_l2 = np.sqrt(np.trapezoid(l2_array,dx=dt))
final_h1 = np.sqrt(np.trapezoid(h1_array,dx=dt))
print("L2 error: ", final_l2)
print("H1 error: ", final_h1)
