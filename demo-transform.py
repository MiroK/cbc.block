"""Demo showing some of the basic block matrix transform functionality."""

from dolfin import *
from block import *
from block.iterative import ConjGrad, LGMRES

mesh = UnitSquare(2,2)
V    = FunctionSpace(mesh, "CG", 1)
v, u = TestFunction(V), TrialFunction(V)

tensor = block_mat([[ 1, 0],
                    [-1, 1]])

print 'Block matrix multiplication: composed object and its result'
I = block_simplify(block_mat([[1, 0],
                              [0, 1]]))
B = block_transpose(tensor)*(tensor+I)
print '\nI =',I
print '\nB =',B
print '\nBx=',block_collapse(B)

A = assemble(u*v*dx)
K = block_kronecker(A,tensor)
b = assemble(v*dx)

print '\n================='
print 'Block matrix multiplication: Kronecker product and its result'
print '\nK =', K
print '\nKx=', block_collapse(K)

print '\n================='
print 'Block matrix multiplication: Inverse of the Kronecker product.'
print 'This can be formed by explicit inversion of the tensor and an'
print 'iterative solver, either blockwise or on the full system.'

import numpy
C,D = K
Di  = block_mat(numpy.linalg.inv(tensor.blocks))
print '\nKinv1=', block_kronecker(Di,ConjGrad(A))
print '\nKinv2=', Di*ConjGrad(C)

AAinv = LGMRES(K, precond=Di)

b0 = b.copy()
bb = block_vec([b, b0])

XX = AAinv*bb

