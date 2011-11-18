from __future__ import division

"""This demo shows the use of a non-trivial block preconditioner for the time
dependent Stokes equations. It is adapted from the code described in the block
preconditioning chapter of the FENiCS book, by Kent-Andre Mardal
<kent-and@simula.no>.

The block structure is as follows,

       | A   B |
  AA = |       |,
       | C   0 |

and the preconditioner to be used is

        | A^  0   |
  BB^ = |         |,
        | 0  L^+M^|

with M defined as the scaled mass matrix and L as the Laplace operator. We use
the ML algebraic multigrid preconditioner for A, L, and M, and the CGN
iterative solver in order to get eigenvalue estimates for the preconditioned
systems.
"""

from dolfin import *
from block import *
from block.iterative import CGN
from block.algebraic.trilinos import ML
import numpy

dolfin.set_log_level(30)

N=2
k_val=1

# Parse command-line arguments like "N=6" or "k_val=0.1"
import sys
for s in sys.argv[1:]:
    exec(s)

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class BoundaryFunction(Expression):
    def value_shape(self):
        return (2,)
    def eval(self, values, x):
        values[0] = 1 if near(x[1],1) else 0
        values[1] = 0

mesh = UnitSquare(N,N)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

f = Constant((0,0))
g = Constant(0)
k = Constant(k_val)
kinv = Constant(1.0/k_val)

v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)

A = assemble((dot(u,v) + k*inner(grad(u),grad(v)))*dx)
B = assemble(div(v)*p*dx)
C = assemble(div(u)*q*dx)
b = assemble(dot(f, v)*dx)

AA = block_mat([[A, B],
                [C, 0]])
bb = block_vec([b, 0])

bc_func = BoundaryFunction()
bc = block_bc([DirichletBC(V, bc_func, Boundary()), None])
bc.apply(AA, bb)

M = assemble(kinv*p*q*dx)
L = assemble(dot(grad(p),grad(q))*dx)

prec = block_mat([[ML(A),      0     ],
                  [0,     ML(L)+ML(M)]])

xx = bb.copy()
xx.randomize()
AAinv = CGN(AA, precond=prec, initial_guess=xx, tolerance=1e-11, show=0)

x = AAinv*bb
e = AAinv.eigenvalue_estimates()

print "N=%d K=%.3g" % (N, sqrt(e[-1]/e[0]))
