"""This demo solves the Stokes equations with block preconditioning.

The original demo is found in demo/undocumented/stokes-taylor-hood/python.

The algebraic system to be solved can be written as

  BB^ AA [sigma u]^T = BB^ [b 0]^T,

where AA is a 2x2 block system with zero in the (2,2) block

       | A   B |
  AA = |       |,
       | C   0 |

and BB^ approximates the inverse of the block operator

       | A   0 |
  BB = |       |.
       | 0   L |
"""

from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from block import *
from block.dolfin_util import *
from block.iterative import *
from block.algebraic.petsc import *

import os


# Load mesh and subdomains
#path = os.path.join(os.path.dirname(__file__), '')
#mesh = Mesh(path+"dolfin-2.xml.gz")
#dim = mesh.topology().dim()
#sub_domains = MeshFunction("size_t", mesh, path+"subdomains.xml.gz")

import sys
n = int(sys.argv[1])
mesh = UnitSquareMesh(n, n)

if MPI.size(mesh.mpi_comm()) > 1:
    print("Stokes demo does not work in parallel because of old-style XML mesh files")
    exit()

sub_domains = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 100)
DomainBoundary().mark(sub_domains, 0)
CompiledSubDomain('near(x[0], 0.)').mark(sub_domains, 1)
CompiledSubDomain('near(x[0], 1.)').mark(sub_domains, 2)

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# No-slip boundary condition for velocity
noslip = Constant((0, 0))
bc0 = DirichletBC(V, noslip, sub_domains, 0)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0"), degree=2)
bc1 = DirichletBC(V, inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
zero = Constant(0)
bc2 = DirichletBC(Q, zero, sub_domains, 2)

# Define variational problem and assemble matrices
v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)

f = Constant((0, 0))

a11 = inner(grad(v), grad(u))*dx
a12 = div(v)*p*dx
a21 = div(u)*q*dx
L1  = inner(v, f)*dx

I  = assemble(p*q*dx)

# Create the block matrix/vector, and apply boundary conditions. A diagonal
# matrix is automatically created to replace the (2,2) block in AA, since bc2
# makes the block non-zero.
bcs = [[bc0, bc1], bc2]
AA = block_assemble([[a11, a12],
                     [a21,  0 ]])
bb  = block_assemble([L1, 0])

block_bc(bcs, True).apply(AA).apply(bb)

# Extract the individual submatrices
[[A, B],
 [C, _]] = AA

# Create preconditioners: An ML preconditioner for A, and the inverse diagonal
# of the mass matrix for the (2,2) block.
Ap = AMG(A)#ML(A, nullspace=rigid_body_modes(V))
Ip = LumpedInvDiag(I)

prec = block_mat([[Ap, 0],
                  [0, Ip]])

# Create the block inverse, using the preconditioned Minimum Residual method
# (suitable for symmetric indefinite problems).
x0 = AA.create_vec()
x0.randomize()

memory = []
#callback = lambda k, n, x, memory=memory: memory.append(n)
AAinv = PETScMinRes(AA, precond=prec, tolerance=1e-6, maxiter=500, relativeconv=True, show=2,
               callback=None,#callback,
               initial_guess=x0)

# Compute solution
u, p = AAinv * bb

print("Norm of velocity coefficient vector: %.15g" % u.norm("l2"))
print("Norm of pressure coefficient vector: %.15g" % p.norm("l2"))

if memory:
    import matplotlib.pyplot as plt
    import numpy as np
    memory = np.array(memory)

    plt.figure()
    plt.semilogy(memory[:, 0], label='u')
    plt.semilogy(memory[:, 1], label='p')
    plt.legend(loc='best')
    plt.show()
