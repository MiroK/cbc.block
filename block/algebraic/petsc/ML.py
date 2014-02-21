from __future__ import division

from block.block_base import block_base

class ML(block_base):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None):
        from petsc4py import PETSc
        from dolfin import info
        from time import time

        T = time()
        Ad = A.down_cast().mat()
        if (nullspace):
            ns = PETSc.NullSpace.create(constant=True, vectors=nullspace, comm=None)
            Ad.setNullSpace(ns)

        self.A = A
        self.ml_prec = PETSc.PC()
        self.ml_prec.create(PETSc.COMM_WORLD)
        self.ml_prec.setType(PETSc.PC.Type.ML)
        self.ml_prec.setOperators(Ad, Ad, PETSc.Mat.Structure.SAME_PRECONDITIONER)
        self.ml_prec.setUp()

        info('constructed ML preconditioner in %.2f s'%(time()-T))

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented
        # apply the ML preconditioner
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
