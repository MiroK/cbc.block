from __future__ import division
from block_base import block_base
from block_mat import block_mat
from block_vec import block_vec

class BlockPrecond_2x2(block_base):
    def __init__(self, op_2x2, reverse=False):
        """In typical use, Ainv and Dinv are preconditioners while B and C are matrices.
        Any of them may also be preconditioners or inner solvers such as ConjGrad, as long
        as each block implements __mul__ or matvec. (B is only used by SGS)."""

        self.op   = op_2x2
        self.idx  = [0,1] if not reverse else [1,0]

class BlockGaussSeidel_2x2(BlockPrecond_2x2):
    def matvec(self, x):
        y = block_vec(2)
        i0,i1 = self.idx
        y[i0] = self.op[i0,i0] * x[i0]
        y[i1] = self.op[i1,i1] * (x[i1] - self.op[i1,i0] * y[i0])
        return y

class BlockSymmetricGaussSeidel_2x2(BlockGaussSeidel_2x2):
    def matvec(self, x):
        y = BlockGaussSeidel_2x2.matvec(self, x)
        i0,i1 = self.idx
        y[i0] -= self.op[i0,i0] * self.op[i0,i1] * y[i1]
        return y

def blockscheme(op_2x2, scheme='jacobi', reverse=False):
    if isinstance(op_2x2, (list, tuple)):
        op_2x2 = block_mat(op_2x2)
    if not isinstance(op_2x2, block_mat) or op_2x2.blocks.shape != (2,2):
        raise TypeError('expected 2x2 block_mat')
    Ainv, B = op_2x2[0,:]
    C, Dinv = op_2x2[1,:]

    if scheme == 'jacobi' or scheme == 'jac':
        Ainv,Dinv = op_2x2[0,0],op_2x2[1,1]
        return block_mat([[Ainv,  0  ],
                              [0,    Dinv]])

    #bGS  = block_mat([[1,  0  ],
    #                      [0, Dinv]]) * block_mat([[ 1,   0],
    #                                                   [-C, 1]]) * block_mat([[Ainv, 0],
    #                                                                              [ 0,   1]])
    if scheme == 'gauss-seidel' or scheme == 'gs':
        #return bGS
        return BlockGaussSeidel_2x2(op_2x2, reverse)

    if scheme == 'symmetric gauss-seidel' or scheme == 'sgs':
        #return block_mat([[1, -Ainv*B],
        #                      [0,  1     ]]) * bGS
        return BlockSymmetricGaussSeidel_2x2(op_2x2, reverse)

    raise TypeError('unknown scheme "%s"'%scheme)
