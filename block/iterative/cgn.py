from __future__ import division
from common import *

def CGN_BABA(B, A, x, b, tolerance=1e-5, relativeconv=False, maxiter=200):
    #####
    # Adapted from code supplied by KAM (Simula PyCC; GPL license),
    #####

    # Is this correct?? Should we not transpose the preconditoner somewhere??

    # jobh 02/2011: Changed residual from sqrt(rho) to sqrt(inner(r,r)) due to negative rho

    # V stands for A^T

    VBr  = 0*x
    VBAp = 0*x

    r     = b - A*x
    Br    = B*r
    A.transpmult(Br,VBr)
    BVBr  = B*VBr


    rho     = inner(BVBr,VBr)
    rho1    = rho
    p       = BVBr.copy()

    iter = 0
    alphas = []
    betas = []
    residuals = [sqrt(inner(r,r))]

    if relativeconv:
        tolerance *= residuals[0]

    while residuals[-1] > tolerance and iter <= maxiter:
        Ap     = A*p
        BAp    = B*Ap;
        A.transpmult(BAp,VBAp)
        alpha  = rho/inner(p,VBAp)
        x      = x + alpha*p
        r      = b-A*x
        Br     = B*r
        A.transpmult(Br,VBr)
        BVBr   = B*VBr

        rho    = inner(BVBr,VBr)
        beta   = rho/rho1
        rho1   = rho
        p      = BVBr+beta*p

        iter  += 1
        alphas.append(alpha)
        betas.append(beta)
        residuals.append(sqrt(inner(r,r)))


    return x, residuals, alphas, betas
