#####
# Original author: Kent Andre Mardal <kent-and@simula.no>
#####

import PyTrilinos
from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.trilinos import *

dolfin.set_log_level(15)

N = 2

# Parse command-line arguments like "N=6"
import sys
for s in sys.argv[1:]:
    print s
    exec(s)

mesh = UnitCube(N,N,N)

V = FunctionSpace(mesh, "N1curl", 1)
Q = FunctionSpace(mesh, "CG", 1)

v,u = TestFunction(V), TrialFunction(V)
q,p = TestFunction(Q), TrialFunction(Q)

A = assemble(dot(u,v)*dx + dot(curl(v), curl(u))*dx)
B = assemble(dot(grad(p),v)*dx)
C = assemble(dot(grad(q),u)*dx)
D = assemble(p*q*dx)

AA = block_mat([[A,  B],
                [C, -D]])
bb = block_vec([0,0])

L = explicit(A+B*InvDiag(D)*C)

prec = block_mat([[ML(L),  0  ],
                  [0,    ML(D)]])

x = AA.create_vec()
x.randomize()
AAinv = CGN(AA, precond=prec, initial_guess=x, tolerance=1e-9, maxiter=2000)

AA * block_vec([0,0])
x = AAinv*bb

print "Number of iterations: ", AAinv.iterations

e = AAinv.eigenvalue_estimates()
print "Sqrt of condition number of BABA: ", sqrt(e[len(e)-1]/e[0])
