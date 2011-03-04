from __future__ import division

import PyTrilinos
from dolfin import *
from block import *
from block.iterative import CGN
from block.algebraic.trilinos import ML
import numpy

N=2
k_val=1

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

L = assemble(kinv*p*q*dx)
M = assemble(dot(grad(p),grad(q))*dx)

prec = block_mat([[ML(A),      0     ],
                  [0,     ML(L)+ML(M)]])

xx = bb.copy()
xx.randomize()
AAinv = CGN(AA, precond=prec, initial_guess=xx, tolerance=1e-11)

x = AAinv*bb
e = AAinv.eigenvalue_estimates()

print "Number of iterations ", AAinv.iterations
print "Sqrt of condition number of BABA  ", sqrt(e[len(e)-1]/e[0])
