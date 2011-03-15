from __future__ import division

from dolfin import Vector
from block.block_base import block_base

class diag_op(block_base):
    def __init__(self, v):
        from PyTrilinos import Epetra
        assert isinstance(v, (Epetra.MultiVector, Epetra.Vector))
        self.v = v

    def matvec(self, b):
        if not isinstance(b, Vector):
            return NotImplemented
        x = Vector(len(b))
        x.down_cast().vec().Multiply(1.0, self.v, b.down_cast().vec(), 0.0)
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
                x = Epetra.Vector(self.v.Map())
                x.Multiply(1.0, self.v, other.vec(), 0.0)
                return diag_op(x)
        except AttributeError:
            raise RuntimeError, "can't extract matrix data from type '%s'"%str(type(other))

    def add(self, other, lscale=1.0, rscale=1.0):
        try:
            from numpy import isscalar
            from PyTrilinos import Epetra
            if isscalar(other):
                x = Epetra.Vector(self.v.Map())
                x.PutScalar(other)
                other = diag_op(x)
            other = other.down_cast()
            if isinstance(other, matrix_op):
                return other.add(self)
            else:
                x = Epetra.Vector(self.v)
                x.Update(rscale, other.vec(), lscale)
                return diag_op(x)
        except AttributeError:
            raise RuntimeError, "can't extract matrix data from type '%s'"%str(type(other))

    def down_cast(self):
        return self
    def vec(self):
        return self.v

    def __str__(self):
        return '<%s %dx%d>'%(self.__class__.__name__,len(self.v),len(self.v))

class matrix_op(block_base):
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

    def add(self, other, lscale=1.0, rscale=1.0):
        try:
            from PyTrilinos import Epetra
            other = other.down_cast()
            if hasattr(other, 'mat'):
                from PyTrilinos import EpetraExt
                C = Epetra.CrsMatrix(Epetra.Copy, self.M.RowMap(), 100)
                assert (0 == EpetraExt.Add(self.M,      False, lscale, C, 0.0))
                assert (0 == EpetraExt.Add(other.mat(), False, rscale, C, 1.0))
                C.FillComplete()
                C.OptimizeStorage()
                return matrix_op(C)
            else:
                raise NotImplementedError, "matrix-diagonal add not implemented (yet?)"
        except AttributeError:
            raise RuntimeError, "can't extract matrix data from type '%s'"%str(type(other))

    def down_cast(self):
        return self
    def mat(self):
        return self.M

    def __str__(self):
        return '<%s %dx%d>'%(self.__class__.__name__,self.M.NumGlobalRows(),self.M.NumGlobalCols())


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

def _explicit(x):
    from block.block_compose import block_compose, block_add, block_sub
    from numpy import isscalar
    from dolfin import Matrix
    if isinstance(x, (matrix_op, diag_op)):
        return x
    elif isinstance(x, Matrix):
        return matrix_op(x.down_cast().mat())
    elif isinstance(x, diag_op):
        return x
    elif isinstance(x, block_compose):
        factors = x.chain[:]
        while len(factors) > 1:
            A = _explicit(factors.pop())
            B = _explicit(factors.pop())
            C = B.matmat(A) if isscalar(A) else A.matmat(B)
            factors.append(C)
        return factors[0]
    elif isinstance(x, block_add):
        A = _explicit(x.A)
        B = _explicit(x.B)
        return B.add(A) if isscalar(A) else A.add(B)
    elif isinstance(x, block_sub):
        A = _explicit(x.A)
        B = _explicit(x.B)
        return B.add(A, lscale=-1.0) if isscalar(A) else A.add(B, rscale=-1.0)
    elif isscalar(x):
        return x
    else:
        raise NotImplementedError, "_explicit for type '%s'"%str(type(x))

def explicit(x):
    from time import time
    from dolfin import info
    T = time()
    res = _explicit(x)
    info('computed explicit matrix representation (%s) in %.2f s'%(str(res),time()-T))
    return res
