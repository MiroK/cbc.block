from __future__ import division

from dolfin import down_cast, Vector
from block.blockbase import blockbase
from block.blockoperator import blockop
import numpy

class LumpedJacobi(blockbase):
    def __init__(self, A):
        from PyTrilinos import Epetra

        A = down_cast(A).mat()
        self.diag = Epetra.Vector(A.RowMap())
        A.InvRowSums(self.diag)

    def matvec(self, b):
        if not isinstance(b, Vector):
            return NotImplemented
        x = Vector(len(b))
        down_cast(x).vec().Multiply(1, self.diag, down_cast(b).vec(), 0)

        return x

class SchurComplement(blockbase):
    """Return the Schur complement of the lower right block."""
    def __init__(self, AA):
        AA = blockop(AA)
        if AA.blocks.shape != (2,2):
            raise TypeError, "SchurComplement: expected 2x2 blocks"

        A,B = AA[0,:]
        C,D = AA[1,:]

        raise RuntimeError, "Not implemented yet!"

del numpy, blockop
