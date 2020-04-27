"""A selection of iterative methods."""

from __future__ import absolute_import
from .iterative import iterative

class ConjGrad(iterative):
    from . import conjgrad
    method = staticmethod(conjgrad.precondconjgrad)

class BiCGStab(iterative):
    from . import bicgstab
    method = staticmethod(bicgstab.precondBiCGStab)

class CGN(iterative):
    from . import cgn
    method = staticmethod(cgn.CGN_BABA)

class SymmLQ(iterative):
    from . import symmlq
    method = staticmethod(symmlq.symmlq)

class TFQMR(iterative):
    from . import tfqmr
    method = staticmethod(tfqmr.tfqmr)

class MinRes(iterative):
    from . import minres
    method = staticmethod(minres.minres)

class SubMinRes(iterative):
    from . import subminres
    method = staticmethod(subminres.subminres)
    
class MinRes2(iterative):
    from . import minres2
    method = staticmethod(minres2.minres)

class PETScMinRes(iterative):
    from . import petscminres
    method = staticmethod(petscminres.petsc_minres)

class LGMRES(iterative):
    from . import lgmres
    __doc__ = lgmres.lgmres.__doc__
    method = staticmethod(lgmres.lgmres)

class Richardson(iterative):
    from . import richardson
    method = staticmethod(richardson.richardson)
