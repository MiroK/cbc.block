from __future__ import division

from block.blockbase import blockbase
from dolfin import warning

class krylovbase(blockbase):
    def __init__(self, A, precond=1.0, tolerance=1e-5, initial_guess=None, name=None, show=1, **kwargs):
        self.B = precond
        self.A = A
        self.initial_guess = initial_guess
        self.show = show
        self.kwargs = kwargs
        if name:
            self.name = name
        self.tolerance = tolerance

    def matvec(self, b):
        if self.initial_guess:
            x = self.initial_guess
        else:
            x = b.copy()
            x.zero()
        try:
            x = self.method(self.B, self.A, x, b, tolerance=self.tolerance, **self.kwargs)
        except Exception, e:
            warning("Error solving " + self.name)
            raise
        x, self.residuals, self.alphas, self.betas = x

        if self.show == 1:
            print '%s: iterations=%d, residual=%.1e' \
                % (self.name, self.iterations, self.residuals[-1])
        elif self.show == 2:
            print '%s: iterations=%d, residual=%.1e, true residual=%.1e' \
                % (self.name, self.iterations, self.residuals[-1], (self.A*x-b).norm('l2'))
        return x

    @property
    def iterations(self):
        return len(self.residuals)-1
    @property
    def converged(self):
        return self.residuals[-1] < self.tolerance

    def eigenvalue_estimates(self):
        #####
        # Adapted from code supplied by KAM (Simula PyCC; GPL license),
        #####

        # eigenvalues estimates in terms of alphas and betas

        import numpy

        n = len(self.alphas)
        A = numpy.zeros([n,n])
        A[0,0] = 1/self.alphas[0]
        for k in range(1, n):
            A[k,k] = 1/self.alphas[k] + self.betas[k-1]/self.alphas[k-1]
            A[k,k-1] = numpy.sqrt(self.betas[k-1])/self.alphas[k-1]
            A[k-1,k] = A[k,k-1]
        e,v = numpy.linalg.eig(A)
        e.sort()
        return e

class ConjGrad(krylovbase):
    name = "ConjGrad"
    import conjgrad
    method = staticmethod(conjgrad.precondconjgrad)

class BiCGStab(krylovbase):
    name = "BiCGStab"
    import bicgstab
    method = staticmethod(bicgstab.precondBiCGStab)

class CGN(krylovbase):
    name = "CGN"
    import cgn
    method = staticmethod(cgn.CGN_BABA)

class SymmLQ(krylovbase):
    name = "SymmLQ"
    import symmlq
    method = staticmethod(symmlq.symmlq)

class TFQMR(krylovbase):
    name = "TFQMR"
    import tfqmr
    method = staticmethod(tfqmr.tfqmr)

class MinRes(krylovbase):
    import minres
    method = staticmethod(minres.minres)

class LGMRES(krylovbase):
    import lgmres
    method = staticmethod(lgmres.lgmres)
