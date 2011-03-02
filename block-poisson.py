from __future__ import division

"""This demo program solves the mixed formulation of Poisson's
equation:

    sigma - grad(u) = 0
         div(sigma) = f

The corresponding weak (variational problem)

    <sigma, tau> + <div(tau), u>   = 0       for all tau
                   <div(sigma), v> = <f, v>  for all v

is solved using BDM (Brezzi-Douglas-Marini) elements of degree k for
(sigma, tau) and DG (discontinuous Galerkin) elements of degree k - 1
for (u, v).

Original implementation: ../cpp/main.cpp by Anders Logg and Marie Rognes
"""

__author__    = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__      = "2007-11-14 -- 2008-12-19"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__   = "GNU LGPL Version 2.1"

# Modified by Marie E. Rognes 2010
# Last changed: 31-08-2010

# Begin demo

import PyTrilinos
from block import *
from block.krylov import *
from block.algebraic import *
from dolfin import *

parameters["linear_algebra_backend"] = "Epetra"

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

# Assemble forms into block matrices, and combine
A = assemble(a11)
B = assemble(a12)
C = assemble(a21)

AA = blockop([[A, B],
              [C, 0]])

b1 = assemble(L2)
b = blockvec([0, b1])

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

#=====================

bc = blockbc([DirichletBC(BDM, G, boundary), None])
bc.apply(AA, b)

Ap = ML(A)

L = assemble(u*v*dx)
Lpre = LumpedJacobi(L)
#Linv = Richardson(L, precond=1e-2, maxiter=10, show=2, tolerance=1e-16, name="Linv")

S = SchurComplement(AA)
Sp = ML(S)

prec = blockop([[Ap, B],
                [C,  Sp]]).scheme('sgs')
#=====================

AAinv = ConjGrad(AA, precond=prec, maxiter=1000, show=2, name='AAinv')


import time
T = -time.time()

#=====================
x = AAinv * b
#=====================

T += time.time()
msg = "%d outer iterations in %.2f seconds" % (AAinv.iterations, T)
if AAinv.converged:
    print "Converged [%s]"%msg
else:
    print "NOT CONVERGED [%s]"%msg


# Plot sigma and u
plot(Function(BDM, x[0]))
plot(Function(DG,  x[1]))

interactive()
