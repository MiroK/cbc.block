from __future__ import division

"""Demo showing the use of the Moore-Penrose pseudoinverse to solve non-square
systems of equations (finding the least squares solution for an overspecified
system, or the minimum-norm / all solutions to an underspecified system.
"""

from __future__ import absolute_import
from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.petsc import *

mesh = UnitSquareMesh(64,64)

V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "CG", 2)

f = Expression("sin(pi*x[0])", degree=5)
u, v = TrialFunction(V), TestFunction(V)
s, t = TrialFunction(W), TestFunction(W)

# Forms with different test/trial functions

a1 = u*t*dx + dot(grad(u), grad(t))*dx
L1 = f*t*dx

a2 = s*v*dx + dot(grad(s), grad(v))*dx
L2 = f*v*dx

# Form and matrix used for preconditioning

am = u*v*dx + dot(grad(u), grad(v))*dx
Ap = assemble(am)

#
# Over-specified system (m > n)
#

# Create linear system (matrices, transpose operator, RHS vector)

A = assemble(a1)
b = assemble(L1)

# Create pseudo-inverse operator (least squares solution)

Apinv = ConjGrad(A.T*A, precond=ML(Ap)**2, show=2) * A.T

# Solve and plot

x = Apinv*b
plot(Function(V, x), title="least squares")

#
# Under-specified system (m < n)
#

# Create linear system (matrices, transpose operator, RHS vector)

A = assemble(a2)
b = assemble(L2)

# Create pseudo-inverse operator (minimum norm solution)

Apinv = A.T * ConjGrad(A*A.T, precond=ML(Ap)**2, show=2)

# Solve and plot

x = Apinv*b
plot(Function(V, x), title="minimum norm")

# All solutions to Ax=b are given as (if A^ is the pseudo-inverse)
#    x = A^ b + (I - A^ A) w
# Here's one alternate solution (of many many):
w = A.create_vec(dim=1)
w[:] = 0.64
x += (1 - Apinv * A) * w
plot(Function(V, x), title="alternate solution")

interactive()
