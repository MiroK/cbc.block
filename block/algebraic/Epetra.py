from __future__ import division

from dolfin import Vector
from block.blockbase import blockbase

class LumpedJacobi(blockbase):
    def __init__(self, A):
        from PyTrilinos import Epetra

        A = A.down_cast().mat()
        self.diag = Epetra.Vector(A.RowMap())
        A.InvRowSums(self.diag)

    def matvec(self, b):
        if not isinstance(b, Vector):
            return NotImplemented
        x = Vector(len(b))
        x.down_cast().vec().Multiply(1, self.diag, b.down_cast().vec(), 0)

        return x

class SchurComplement(blockbase):
    """Return the Schur complement of the (2,2) block. A diagonal approximation
    is used for the inverse of the (1,1) block."""
    def __init__(self, AA):
        from PyTrilinos import Epetra, EpetraExt
        import numpy
        from block.blockoperator import blockop

        AA = blockop(AA)
        if AA.blocks.shape != (2,2):
            raise TypeError, "SchurComplement: expected 2x2 blocks"

        A = AA[0,0].down_cast().mat()
        B = AA[0,1].down_cast().mat()
        C = AA[1,0].down_cast().mat()
        D = AA[1,1] # may be scalar (0)

        Adiag = Epetra.Vector(A.RowMap())
        A.ExtractDiagonalCopy(Adiag)
        Adiag.Reciprocal(Adiag)

        S = Epetra.CrsMatrix(Epetra.Copy, C.RowMap(), 100)

        # Make a scaled copy of C
        C = Epetra.CrsMatrix(C)
        C.LeftScale(Adiag)

        assert (0 == EpetraExt.Multiply(C, False, B, False, S))
        if numpy.isscalar(D):
            if D != 0:
                # Extract diagonal of S
                # Add D to diagonal
                # Put diagonal back
                raise NotImplementedError, 'SchurComplement: The (2,2) block must be a matrix or scalar 0'
        else:
            assert (0 == EpetraExt.Add(D.down_cast().mat(), False, -1, S, 1))

        S.FillComplete()
        S.OptimizeStorage()
        self.S = S

    # For algebraic preconditioners --- they call A.down_cast().mat() to access Epetra matrix

    def down_cast(self):
        return self

    def mat(self):
        return self.S

    def matvec(self, b):
        if not isinstance(b, Vector):
            return NotImplemented
        x = Vector(len(b))
        self.S.Apply(b.down_cast().vec(), x.down_cast().vec())
        return x
