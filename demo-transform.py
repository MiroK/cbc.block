"""Demo showing some of the basic block matrix transform functionality."""

from dolfin import *
from block import *
from block.iterative import ConjGrad

mesh = UnitSquare(2,2)

V    = FunctionSpace(mesh, "CG", 1)
v, u = TestFunction(V), TrialFunction(V)

tensor1 = block_mat([[1, 0],
                     [0, 1]])
tensor2 = block_mat([[1,-1],
                     [0, 1]])

print 'Block matrix multiplication: composed object and its result'
I = block_simplify(tensor1)
B = block_transpose(tensor1)*(tensor1+I)
print '\nI =',I
print '\nB =',B
print '\nBx=',block_collapse(B)


M = assemble(u*v*dx)
A = assemble(dot(grad(u),grad(v))*dx)

K = block_kronecker(A,tensor1)
L = block_kronecker(M,tensor2)

AA = K + L


print '\n================='
print 'Block matrix multiplication: Kronecker product and its result'
print '\nK =', AA
print '\nKx=', block_collapse(AA)

print '\n================='
print 'Block matrix multiplication: Inverse of the Kronecker product.'
print 'This can be formed by explicit inversion of the tensor and an'
print 'iterative solver, either blockwise or on the full system.'

import numpy
C,D = AA
Di  = block_mat(numpy.linalg.inv(tensor1.blocks))
print '\nKinv1=', block_kronecker(Di,ConjGrad(M+A))
print '\nKinv2=', Di*ConjGrad(C)
