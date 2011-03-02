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

class MatMult(blockbase):
    def __init__(self, A, B, scale=None):
        from PyTrilinos import Epetra, EpetraExt

        A = A.down_cast().mat()
        B = B.down_cast().mat()

        if scale is not None:
            B = Epetra.CrsMatrix(B)
            B.LeftScale(scale)

        C = Epetra.CrsMatrix(Epetra.Copy, A.RowMap(), 100)
        assert (0 == EpetraExt.Multiply(A, False, B, False, C))
        C.OptimizeStorage()

        self.C = C

    # For algebraic preconditioners --- they call A.down_cast().mat() to access Epetra matrix
    def down_cast(self):
        return self
    def mat(self):
        return self.C

    def matvec(self, b):
        if not isinstance(b, Vector):
            return NotImplemented
        x = Vector(len(b))
        self.C.Apply(b.down_cast().vec(), x.down_cast().vec())
        return x

class SchurComplement(MatMult):
    """Return the Schur complement of the (2,2) block. A diagonal approximation
    is used for the inverse of the (1,1) block.

    C * inv(diag(A)) * B - D"""
    def __init__(self, AA):
        from PyTrilinos import Epetra
        import numpy
        from block.blockoperator import blockop

        AA = blockop(AA)
        if AA.blocks.shape != (2,2):
            raise TypeError, "SchurComplement: expected 2x2 blocks"

        A,B = AA[0,:]
        C,D = AA[1,:]

        Adiag = Epetra.Vector(A.down_cast().mat().RowMap())
        A.down_cast().mat().ExtractDiagonalCopy(Adiag)
        Adiag.Reciprocal(Adiag)

        MatMult.__init__(self, C, B, scale=Adiag)

        if numpy.isscalar(D):
            if D != 0:
                # Extract diagonal of S
                # Add D to diagonal
                # Put diagonal back
                raise NotImplementedError, 'SchurComplement: The (2,2) block must be a matrix or scalar 0'
        else:
            assert (0 == EpetraExt.Add(D.down_cast().mat(), False, -1, self.C, 1))
