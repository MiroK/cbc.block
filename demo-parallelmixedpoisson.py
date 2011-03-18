from __future__ import division

"""This demo program shows the use of block preconditioners for Mixed Poisson
in a parallel setting. The general description of the block system can be found
in demo-mixedpoisson.py.

The main limitation that is worked around here is the lack of ability to set
Dirichlet boundary conditions in a symmetric way in parallel. This limitation
may be lifted eventually, but in the meantime this demo shows some workarounds.
In particular, we use a nonsymmetric iterative solver, and we perform a few
tricks to make the preconditioner symmetric.

To recap, the system can be written as

       | A   B |
  AA = |       |,
       | C   0 |

and the preconditioner as

        | A^  0 |
  BB^ = |       |,
        | 0   L^|

where L is the Laplace operator, formed as L=C*B.

The main points are:
1) To make A symmetric, we use assemble_system() rather than assemble().
2) Zero the BC rows of B using DirichletBC.zero().
3) To make L symmetric, create it using block_transpose(B)*B instead of C*B.
4) Use TFQMR (or BiCGStab) rather than MinRes (or SymmLQ) solver.
"""

# Since we use ML from Trilinos, we must import PyTrilinos before any dolfin
# modules. This works around a bug with MPI initialisation/destruction order.
import PyTrilinos

from block import *
from block.iterative import TFQMR
from block.algebraic.trilinos import ML, explicit
from dolfin import *

# Create mesh
mesh = UnitSquare(32,32)

# Define function spaces
BDM = FunctionSpace(mesh, "BDM", 1)
DG = FunctionSpace(mesh, "DG", 0)

# Define trial and test functions
tau, sigma = TestFunction(BDM), TrialFunction(BDM)
v,   u     = TestFunction(DG),  TrialFunction(DG)

# Define source function
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")

# Define variational forms (one per block)
a11 = dot(sigma, tau) * dx
a12 = div(tau) * u * dx
a21 = div(sigma) * v *dx
L2  = - f * v * dx

# Define function G such that G \cdot n = g
class BoundarySource(Expression):
    def __init__(self, mesh):
        self.mesh = mesh
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        g = sin(5*x[0])
        values[0] = g*n[0]
        values[1] = g*n[1]
    def value_shape(self):
        return (2,)

G = BoundarySource(mesh)

# Define essential boundary
def boundary(x, on_boundary):
    return on_boundary and near(x[1], 0) or near(x[1], 1)

# Define the boundary condition for the BDM space (sigma)
BDM_bc = DirichletBC(BDM, G, boundary)

# Assemble forms into block matrices, and combine
A,b0 = assemble_system(a11, tau[0]*dx, BDM_bc) # symmetric BC modification
B = assemble(a12)
BDM_bc.zero(B) # zero BC rows
C = assemble(a21)
b1 = assemble(L2)

AA = block_mat([[A, B],
                [C, 0]])

b = block_vec([b0, b1])

# Create a preconditioner for A (using the ML preconditioner from Trilinos)
Ap = ML(A)

# Create an ML preconditioner for L. Use block_transpose(B) rather than C,
# because the result is then symmetric. ML requires access to the matrix
# elements, so we use explicit() to perform the matrix-matrix multiplication.
L = explicit(block_transpose(B)*B)
Lp = ML(L)

# Define the block preconditioner
AAp = block_mat([[Ap, 0],
                 [0,  Lp]])

# Define the block inverse using the Transpose Free Quasi Minimum Residual
# method, suitable for nonsymmetric problems.
AAinv = TFQMR(AA, precond=AAp, show=2, name='AA^')

#=====================
# Solve the system
Sigma, U = AAinv * b
#=====================

# Plot sigma and u
plot(Function(BDM, Sigma))
plot(Function(DG,  U))

interactive()
