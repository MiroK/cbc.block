from __future__ import division

from dolfin import Vector
from block.blockbase import blockbase

class diag_op(blockbase):
    def __init__(self, v):
        from PyTrilinos import Epetra
        assert isinstance(v, (Epetra.MultiVector, Epetra.Vector))
        self.v = v

    def matvec(self, b):
        if not isinstance(b, Vector):
            return NotImplemented
        x = Vector(len(b))
        x.down_cast().vec().Multiply(1, self.v, b.down_cast().vec(), 0)
        return x

    def matmat(self, other):
        try:
            from numpy import isscalar
            from PyTrilinos import Epetra
            if isscalar(other):
                x = Epetra.Vector(self.v)
                x.Scale(other)
                return diag_op(other)
            other = other.down_cast()
            if hasattr(other, 'mat'):
                C = Epetra.CrsMatrix(other.mat())
                C.LeftScale(self.v)
                return matrix_op(C)
            else:
                x = Epetra.Vector(other.vec().EpetraMap())
                x.Multiply(1, self.v, other.vec(), 0)
                return diag_op(x)
        except AttributeError:
            raise RuntimeError, "can't extract matrix data from type '%s'"%str(type(other))

    def down_cast(self):
        return self
    def vec(self):
        return self.v

class matrix_op(blockbase):
    def __init__(self, M):
        from PyTrilinos import Epetra
        assert isinstance(M, (Epetra.CrsMatrix, Epetra.FECrsMatrix))
        self.M = M

    def matvec(self, b):
        if not isinstance(b, Vector):
            return NotImplemented
        x = Vector(len(b))
        self.M.Apply(b.down_cast().vec(), x.down_cast().vec())
        return x

    def matmat(self, other):
        try:
            from numpy import isscalar
            from PyTrilinos import Epetra
            if isscalar(other):
                C = Epetra.CrsMatrix(self.M)
                C.Scale(other)
                return matrix_op(C)
            other = other.down_cast()
            if hasattr(other, 'mat'):
                from PyTrilinos import EpetraExt
                C = Epetra.CrsMatrix(Epetra.Copy, self.M.RowMap(), 100)
                assert (0 == EpetraExt.Multiply(self.M, False, other.mat(), False, C))
                C.OptimizeStorage()
                return matrix_op(C)
            else:
                C = Epetra.CrsMatrix(self.M)
                C.RightScale(other.vec())
                return matrix_op(C)
        except AttributeError:
            raise RuntimeError, "can't extract matrix data from type '%s'"%str(type(other))

    def down_cast(self):
        return self
    def mat(self):
        return self.M

class Diag(diag_op):
    def __init__(self, A):
        from PyTrilinos import Epetra
        A = A.down_cast().mat()
        v = Epetra.Vector(A.RowMap())
        A.ExtractDiagonalCopy(v)
        diag_op.__init__(self, v)

class InvDiag(Diag):
    def __init__(self, A):
        Diag.__init__(self, A)
        self.v.Reciprocal(self.v)

class LumpedInvDiag(diag_op):
    def __init__(self, A):
        from PyTrilinos import Epetra
        A = A.down_cast().mat()
        v = Epetra.Vector(A.RowMap())
        A.InvRowSums(v)
        diag_op.__init__(self, v)

class explicit(matrix_op):
    def __init__(self, x):
        from block.blockcompose import blockcompose
        from numpy import isscalar
        from dolfin import Matrix
        if isinstance(x, blockcompose):
            factors = x.chain
            while len(factors) > 1:
                A = factors.pop()
                B = factors.pop()
                if isinstance(A, Matrix):
                    _A = A # postpone garbage collection
                    A = matrix_op(A.down_cast().mat())
                if isinstance(B, Matrix):
                    _B = B # postpone garbage collection
                    B = matrix_op(B.down_cast().mat())
                if isscalar(A):
                    C = B.matmat(A)
                else:
                    C = A.matmat(B)
                factors.append(C)
            matrix_op.__init__(self, factors[0].down_cast().mat())
        else:
            raise NotImplementedError, "explicit for type '%s'"%str(type(x))
