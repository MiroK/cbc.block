"""This demo solves the Stokes equations with block preconditioning.

The original demo is found in demo/undocumented/stokes-taylor-hood/python.

The algebraic system to be solved can be written as

  BB^ AA [sigma u]^T = BB^ [0 b]^T,

where AA is a 2x2 block system with zero in the (2,2) block

       | A   B |
  AA = |       |,
       | C   0 |

and BB^ approximates the inverse of the block operator

       | A   0 |
  BB = |       |.
       | 0   L |
"""

# Since we use ML from Trilinos, we must import PyTrilinos before any dolfin
# modules. This works around a bug with MPI initialisation/destruction order.
import PyTrilinos

from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.trilinos import *

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

A  = assemble(inner(grad(v), grad(u))*dx)
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

# Create preconditioners: An ML preconditioner for A, and the ML inverse of the
# mass matrix for the (2,2) block.
Ap = ML(A)
Ip = ML(I)

prec = block_mat([[Ap, 0],
                  [0, Ip]])

# Create the block inverse, using the preconditioned Minimum Residual method
# (suitable for symmetric indefinite problems).
AAinv = MinRes(AA, precond=prec, tolerance=1e-10, maxiter=500, show=2)

# Compute solution
u, p = AAinv * b

print "Norm of velocity coefficient vector: %.15g" % u.norm("l2")
print "Norm of pressure coefficient vector: %.15g" % p.norm("l2")

# Plot solution
plot(Function(V, u))
plot(Function(Q, p))
interactive()
