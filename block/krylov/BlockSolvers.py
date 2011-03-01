from __future__ import division

from BlockStuff import BlockBase
import Krylov
import numpy

class KrylovBase(BlockBase):
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

class ConjGrad(KrylovBase):
    name = "ConjGrad"
    method = staticmethod(Krylov.precondconjgrad)

class BiCGStab(KrylovBase):
    name = "BiCGStab"
    method = staticmethod(Krylov.precondBiCGStab)

class CGN(KrylovBase):
    name = "CGN"
    method = staticmethod(Krylov.CGN_BABA)

class SymmLQ(KrylovBase):
    name = "SymmLQ"
    method = staticmethod(Krylov.symmlq)

class TFQMR(KrylovBase):
    name = "TFQMR"
    method = staticmethod(Krylov.tfqmr)
