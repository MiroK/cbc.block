"""Demo showing some of the basic block matrix transform functionality."""

from dolfin import *
from block import *

mesh = UnitSquare(2,2)
V    = FunctionSpace(mesh, "CG", 1)
v, u = TestFunction(V), TrialFunction(V)

tensor = block_mat([[1,-1],
                    [0, 1]])

print 'Block matrix multiplication: composed object and its result'
I = simplify(block_mat([[1, 0],
                        [0, 1]]))
B = block_transpose(tensor)*(tensor+I)
print '\nI =',I
print '\nB =',B
print '\nBx=',inside_out(B)

A = assemble(u*v*dx)
K = kronecker(A,tensor)

print '\n================='
print 'Block matrix multiplication: Kronecker product and its result'
print '\nK =', K
print '\nKx=', inside_out(K)
