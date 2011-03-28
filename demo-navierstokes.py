"""This demo solves the Navier-Stokes equations with block preconditioning.

NOTE: Needs work. Must define a better precond, this one doesn't work for smaller mu.

It is a modification of the Stokes demo (demo-stokes.py).
"""

# Since we use ML from Trilinos, we must import PyTrilinos before any dolfin
# modules. This works around a bug with MPI initialisation/destruction order.
# Furthermore, scipy (in LGMRES) seems to crash unless it's loaded before
# PyTrilinos.
import scipy
import PyTrilinos

from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.trilinos import *

dolfin.set_log_level(15)

# Load mesh and subdomains
mesh = Mesh("dolfin-2.xml.gz")
sub_domains = MeshFunction("uint", mesh, "subdomains.xml.gz")

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# No-slip boundary condition for velocity
noslip = Constant((0, 0))
bc0 = DirichletBC(V, noslip, sub_domains, 0)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0"))
bc1 = DirichletBC(V, inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
zero = Constant(0)
bc2 = DirichletBC(Q, zero, sub_domains, 2)

# Define variational problem and assemble matrices
v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)

f = Constant((0, 0))
w = Constant((1, 0))
mu = Constant(1e-2)

A  = assemble(mu*inner(grad(v), grad(u))*dx + inner(dot(grad(u),w),v)*dx)
B  = assemble(div(v)*p*dx)
C  = assemble(div(u)*q*dx)
I  = assemble(p*q*dx)
b0 = assemble(inner(v, f)*dx)

# Create the block matrix/vector. We need a matrix in the (2,2) block instead
# of just zero, because it must be modified for the Dirichlet boundary
# conditions. At the moment there is no simple way to create a diagonal matrix
# in Dolfin, so we use the "trick" of right-multiplying the suitably sized mass
# matrix by zero. (Note that 0*I would not work, since that creates a
# composed operator instead of a matrix.)
AA = block_mat([[A, B],
                [C, I*0]])
b  = block_vec([b0, 0])

# Apply boundary conditions
bcs = block_bc([[bc0, bc1], [bc2]])
bcs.apply(AA, b)

# Create preconditioners: An ILU preconditioner for A, and an ML inverse of the
# Schur complement approximation for the (2,2) block.
Ap = ILU(A)
Dp = ML(collapse(C*InvDiag(A)*B))

prec = block_mat([[Ap, B],
                  [C, -Dp]]).scheme('sgs')

# Create the block inverse, using the LGMRES method (suitable for general problems).
AAinv = LGMRES(AA, precond=prec, tolerance=1e-5, maxiter=50, show=2)

# Compute solution
u, p = AAinv * b

print "Norm of velocity coefficient vector: %.15g" % u.norm("l2")
print "Norm of pressure coefficient vector: %.15g" % p.norm("l2")

# Plot solution
plot(Function(V, u))
plot(Function(Q, p))
interactive()
