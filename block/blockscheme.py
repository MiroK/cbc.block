class BlockPrecond_2x2(BlockBase):
    def __init__(self, op_2x2, reverse=False):
        """In typical use, Ainv and Dinv are preconditioners while B and C are matrices.
        Any of them may also be BlockOperators, or inner solvers such as ConjGrad, as long
        as each block implements __mul__ or matvec; thus, any number of blocks may be preconditioned
        using nested 2x2 blocks. (B is only used by SGS)."""

        self.op   = op_2x2
        self.idx  = [0,1] if not reverse else [1,0]

class BlockGaussSeidel_2x2(BlockPrecond_2x2):
    def matvec(self, x):
        y = BlockVector(2)
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

def BlockScheme(op_2x2, scheme='jacobi', reverse=False):
    if isinstance(op_2x2, (list, tuple)):
        op_2x2 = BlockOperator(op_2x2)
    if not isinstance(op_2x2, BlockOperator) or op_2x2.blocks.shape != (2,2):
        raise TypeError('expected 2x2 BlockOperator')
    Ainv, B = op_2x2[0,:]
    C, Dinv = op_2x2[1,:]

    if scheme == 'jacobi' or scheme == 'jac':
        Ainv,Dinv = op_2x2[0,0],op_2x2[1,1]
        return BlockOperator([[Ainv,  0  ],
                              [0,    Dinv]])

    #bGS  = BlockOperator([[1,  0  ],
    #                      [0, Dinv]]) * BlockOperator([[ 1,   0],
    #                                                   [-C, 1]]) * BlockOperator([[Ainv, 0],
    #                                                                              [ 0,   1]])
    if scheme == 'gauss-seidel' or scheme == 'gs':
        #return bGS
        return BlockGaussSeidel_2x2(op_2x2, reverse)

    if scheme == 'symmetric gauss-seidel' or scheme == 'sgs':
        #return BlockOperator([[1, -Ainv*B],
        #                      [0,  1     ]]) * bGS
        return BlockSymmetricGaussSeidel_2x2(op_2x2, reverse)

    raise TypeError('unknown scheme "%s"'%scheme)
