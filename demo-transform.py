"""Demo showing some of the basic block matrix transform functionality."""

from dolfin import *
from block import *
from block.iterative import ConjGrad

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

print '\n================='
print 'Block matrix multiplication: Inverse of the Kronecker product.'
print 'This can be formed by explicit inversion of the tensor and an'
print 'iterative solver, either blockwise or on the full system.'

import numpy
C,D = K
Di  = block_mat(numpy.linalg.inv(tensor.blocks))
print '\nKinv1=', kronecker(Di,ConjGrad(A))
print '\nKinv2=', Di*ConjGrad(C)
