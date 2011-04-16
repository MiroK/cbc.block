from __future__ import division

"""This demo program shows the use of block preconditioners for Mixed Poisson
in a parallel setting. The general description of the block system can be found
in demo-mixedpoisson.py.

The main limitation that is worked around here is the lack of ability to set
Dirichlet boundary conditions in a symmetric way in parallel. This limitation
may be lifted eventually, but in the meantime this demo shows some workarounds.

To recap, the system can be written as

       | A   B |
  AA = |       |,
       | C   0 |

where C=B^T before boundary conditions are applied, and the preconditioner as

        | A^  0 |
  BB^ = |       |,
        | 0   L^|

where L is the Laplace operator, formed as L=C*B.

The main points are:
1) To make A symmetric, we use assemble_system() rather than assemble().
2) Zero the BC rows of B using DirichletBC.zero().
3) To make L symmetric, create it using block_transpose(B)*B instead of C*B.
4) To make AA symmetric, replace C with block_transpose(B) and modify the RHS
5) Plotting doesn't work in parallel with trilinos backend, so skip that.

Alternatively, a non-symmetric solver such as TFQMR or BiCGStab may be used.
"""

# Since we use ML from Trilinos, we must import PyTrilinos before any dolfin
# modules. This works around a bug with MPI initialisation/destruction order.
import PyTrilinos

from block import *
from block.iterative import TFQMR, Richardson, ConjGrad, MinRes
from block.algebraic.trilinos import ML, collapse
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

 # Symmetric BC modification for the A block. (Is there a simpler way to
 # specify a zero RHS?)
A,b0 = assemble_system(a11, Constant(0)*tau[0]*dx, BDM_bc)
# Zero the BC rows of the B block.
B = assemble(a12)
BDM_bc.zero(B)

# Assemble the second block-row normally.
C = assemble(a21)
b1 = assemble(L2)

# Replace C with B^T in the block matrix
BT = block_transpose(B)
AA = block_mat([[A,  B],
                [BT, 0]])

# Notice that C = B^T + (C-B^T). We assume that the diagonal entries of A are 1
# for the Dirichlet boundary conditions, and hence that x=b0 for the boundary
# unknowns. Since the only non-zero columns of C-B^T are those associated with
# boundary conditions, (C-B^T)*x = (C-B^T)*b0, and thus B^T*x = b1-(C-B^T)*y =
# b1-(C-B^T)*b0.

b = block_vec([b0, b1-(C-BT)*b0])

# Create a preconditioner for A (using the ML preconditioner from Trilinos)
Ap = ML(A)

# Create an approximate inverse of L=C*B using inner ConjGrad
# iterations. ConjGrad is not completely safe as inner solver -- see comment in
# demo-mixedpoisson.
L = BT*B
Lp = ConjGrad(L, iter=40, name='L^')

# Define the block preconditioner
AAp = block_mat([[Ap, 0],
                 [0,  Lp]])

# Define the block inverse using an outer Preconditioned Minimum Residual
# method, suitable for symmetric indefinite problems.
AAinv = MinRes(AA, precond=AAp, show=2, name='AA^')

#=====================
# Solve the system
Sigma, U = AAinv * b
#=====================

# Print norms that can be compared with those reported by demo-mixedpoisson
print 'norm Sigma:', Sigma.norm('l2')
print 'norm U    :', U.norm('l2')

# Plotting doesn't seem to work in parallel with the Epetra backend
#plot(Function(BDM, Sigma))
#plot(Function(DG,  U))

#interactive()
