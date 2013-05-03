from __future__ import division
from common import *

def precondBiCGStab(B, A, x, b, tolerance, maxiter, progress, relativeconv=False, callback=None):
    #####
    # Adapted from code supplied by KAM (Simula PyCC; GPL license). This code
    # relicensed under LGPL v2.1 or later, in agreement with the original
    # authors:
    # Kent Andre Mardal <kent-and@simula.no>
    # Ola Skavhaug <skavhaug@simula.no>
    # Gunnar Staff <gunnaran@simula.no>
    #####

    r = b - A*x

    p  = r.copy()
    r0 = r.copy()
    rr0 = inner(r,r0)

    iter = 0
    alphas = []
    betas = []
    residuals = [sqrt(rr0)]

    if relativeconv:
        tolerance *= residuals[0]

    while residuals[-1] > tolerance and iter < maxiter:
        Bp    = B*p
        ABp   = A*Bp
        alpha = rr0/inner(r0,ABp)
        s     = r-alpha*ABp
        Bs    = B*s
        ABs   = A*Bs
        w     = inner(ABs,s)/inner(ABs,ABs)
        x    += alpha*Bp+w*Bs
        r     = s - w*ABs
        rrn   = inner(r,r0)

        residual = sqrt(inner(r,r))

        # Call user provided callback with solution
        if callable(callback):
            callback(k=iter, x=x, r=residual)

        beta  = (rrn/rr0)*(alpha/w)
        if beta==0.0:
            print "BiCGStab breakdown, beta=0, at iter=",iter," with residual=", residual
            return x, residuals, alphas, betas
        rr0   = rrn
        p     = r+beta*(p-w*ABp)

        iter += 1
        progress += 1
        alphas.append(alpha)
        betas.append(beta)
        residuals.append(residual)

    return x, residuals, alphas, betas
