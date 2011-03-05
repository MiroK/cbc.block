from __future__ import division

from block.block_base import block_base
class iterative(block_base):
    def __init__(self, A, precond=1.0, tolerance=1e-5, initial_guess=None, name=None, show=1, iter=None, **kwargs):
        self.B = precond
        self.A = A
        self.initial_guess = initial_guess
        self.show = show
        self.kwargs = kwargs
        self.name = name if name else self.__class__.__name__
        self.tolerance = tolerance
        if iter is not None:
            self.tolerance = 0
            kwargs['maxiter'] = iter

    def matvec(self, b):
        from time import time
        from block.block_vec import block_vec

        T = time()

        # If x and initial_guess are block_vecs, some of the blocks may be
        # scalars (although block_bc.apply() converts these to vectors, so
        # normally they are not). To be sure, call allocate() on them.

        if isinstance(b, block_vec):
            b.allocate(self.A)

        if self.initial_guess:
            x = self.initial_guess
            if isinstance(x, block_vec):
                x.allocate(self.A)
        else:
            x = b.copy()
            x.zero()
        try:
            x = self.method(self.B, self.A, x, b, tolerance=self.tolerance, **self.kwargs)
        except Exception, e:
            from dolfin import warning
            warning("Error solving " + self.name)
            raise
        x, self.residuals, self.alphas, self.betas = x

        if self.tolerance == 0:
            msg = "Done"
        elif self.converged:
            msg = "Converged"
        else:
            msg = "NOT CONV."

        if self.show == 1:
            print '%s: %s [iter=%2d, time=%.2fs, res=%.1e]' \
                % (self.name, msg, self.iterations, time()-T, self.residuals[-1])
        elif self.show == 2:
            print '%s: %s [iter=%2d, time=%.2fs, res=%.1e, true res=%.1e]' \
                % (self.name, msg, self.iterations, time()-T, self.residuals[-1], (self.A*x-b).norm('l2'))
        return x

    @property
    def iterations(self):
        return len(self.residuals)-1
    @property
    def converged(self):
        return self.tolerance == 0 or self.residuals[-1] < self.tolerance

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

class ConjGrad(iterative):
    import conjgrad
    method = staticmethod(conjgrad.precondconjgrad)

class BiCGStab(iterative):
    import bicgstab
    method = staticmethod(bicgstab.precondBiCGStab)

class CGN(iterative):
    import cgn
    method = staticmethod(cgn.CGN_BABA)

class SymmLQ(iterative):
    import symmlq
    method = staticmethod(symmlq.symmlq)

class TFQMR(iterative):
    import tfqmr
    method = staticmethod(tfqmr.tfqmr)

class MinRes(iterative):
    import minres
    method = staticmethod(minres.minres)

class LGMRES(iterative):
    import lgmres
    method = staticmethod(lgmres.lgmres)

class Richardson(iterative):
    import richardson
    method = staticmethod(richardson.richardson)

del iterative, block_base
