from __future__ import division
from common import *

def precondconjgrad(B, A, x, b, tolerance=1e-5, relativeconv=False, maxiter=300):
    #####
    # Adapted from code supplied by KAM (Simula PyCC; GPL license),
    #####

    r = b - A*x
    z = B*r
    d = z
    rz = inner(r,z)
    if rz < 0:
        raise ValueError('Matrix is not positive')

    iter = 0
    alphas = []
    betas = []
    residuals = [sqrt(rz)]

    if relativeconv:
        tolerance *= residuals[0]

    while residuals[-1] > tolerance and iter <= maxiter:
        z = A*d
        dz = inner(d,z)
        if dz == 0:
            print 'ConjGrad breakdown'
            break
        alpha = rz/dz
        x += alpha*d
        r -= alpha*z
        z = B*r
        rz_prev = rz
        rz = inner(r,z)
        if rz < 0:
            print 'ConjGrad breakdown'
            # Restore pre-breakdown state. Don't know if it helps any, but it's
            # consistent with returned quasi-residuals.
            x -= alpha*d
            break
        beta = rz/rz_prev
        d = z + beta*d

        iter += 1
        alphas.append(alpha)
        betas.append(beta)
        residuals.append(sqrt(rz))

    return x, residuals, alphas, betas
