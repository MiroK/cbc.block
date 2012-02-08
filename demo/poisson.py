from block import *
from block.iterative import *
from block.algebraic.trilinos import *
from dolfin import *
from block.dolfin_util import *
import numpy

# Function spaces, elements

mesh = UnitSquare(16,16)

V = FunctionSpace(mesh, "CG", 1)

f = Expression("sin(3.14*x[0])")
u, v =  TrialFunction(V), TestFunction(V)

a = u*v*dx + dot(grad(u), grad(v))*dx  
L = f*v*dx 

A = assemble(a)
b = assemble(L)

B = ML(A)

Ainv = ConjGrad(A, precond=B, tolerance=1e-10, show=3) 

x = Ainv*b 

u = Function(V)
u.vector()[:] = x[:]
plot(u, title="u")

u2 = Function(V)
solve(A, u2.vector(), b)
plot(u2, title="u2")

interactive()

