from __future__ import division

from block.block_base import block_base
from petsc4py import PETSc

class precond(block_base):
    def __init__(self, A, prectype, parameters=None, pdes=1, nullspace=None):
        from dolfin import info
        from time import time

        T = time()
        Ad = A.down_cast().mat()

        # A bug somewhere... setUp() below crashes with BS>1 matrices
        if prectype == 'ml' and Ad.getBlockSize() > 1:
            Ad = PETSc.Mat().createAIJ(Ad.getSize(), csr=Ad.getValuesCSR())

        if nullspace:
            nullspace = [v.down_cast().vec().copy() for v in nullspace]
            ns = PETSc.NullSpace()
            ns.create(constant=False, vectors=nullspace)
            Ad.setNullSpace(ns)

        self.A = A
        self.ml_prec = PETSc.PC()
        self.ml_prec.create(PETSc.COMM_WORLD)
        self.ml_prec.setType(prectype)
        self.ml_prec.setOperators(Ad, Ad, PETSc.Mat.Structure.SAME_PRECONDITIONER)
        self.ml_prec.setUp()

        info('constructed %s preconditioner in %.2f s'%(self.__class__.__name__, time()-T))

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented
        x = self.A.create_vec(dim=1)
        if len(x) != len(b):
            raise RuntimeError(
                'incompatible dimensions for PETSc matvec, %d != %d'%(len(x),len(b)))

        self.ml_prec.apply(b.down_cast().vec(), x.down_cast().vec())
        return x

    def down_cast(self):
        return self.ml_prec

    def __str__(self):
        return '<%s prec of %s>'%(self.__class__.__name__, str(self.A))

class ML(precond):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None):
        precond.__init__(self, A, PETSc.PC.Type.ML, parameters, pdes, nullspace)

class ILU(precond):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None):
        precond.__init__(self, A, PETSc.PC.Type.ILU, parameters, pdes, nullspace)
