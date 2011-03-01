from __future__ import division
from math import sqrt
import numpy

def inner(x,y):
    return x.inner(y)

def norm(v):
    return v.norm('l2')

eps = numpy.finfo(float).eps

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

def precondBiCGStab(B, A, x, b, tolerance=1e-05, relativeconv=False, maxiter=200):
    #####
    # Adapted from code supplied by KAM (Simula PyCC; GPL license),
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

    while residuals[-1] > tolerance and iter <= maxiter:
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
        beta  = (rrn/rr0)*(alpha/w)
        if beta==0.0:
            print "BiCGStab breakdown, beta=0, at iter=",iter," with residual=",sqrt(inner(r,r))
            return x, residuals
        rr0   = rrn
        p     = r+beta*(p-w*ABp)

        iter += 1
        alphas.append(alpha)
        betas.append(beta)
        residuals.append(sqrt(inner(r,r)))

    return x, residuals, alphas, betas

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

        rho1   = rho
        p      = BVBr+beta*p

        iter  += 1
        alphas.append(alpha)
        betas.append(beta)
        residuals.append(sqrt(inner(r,r)))


    return x, residuals, alphas, betas


def symmlq(B, A, x, b, tolerance=1e-5, relativeconv=False, maxiter=200, shift=0):
    #####
    # Adapted from PyKrylov (https://github.com/dpo/pykrylov; LGPL license)
    #####

    msg={
        -1:' beta2 = 0.  If M = I, b and x are eigenvectors',
         0:' beta1 = 0.  The exact solution is  x = 0',
         1:' Requested accuracy achieved, as determined by tolerance',
         2:' Reasonable accuracy achieved, given eps',
         3:' x has converged to an eigenvector',
         4:' acond has exceeded 0.1/eps',
         5:' The iteration limit was reached',
         6:' aprod  does not define a symmetric matrix',
         7:' msolve does not define a symmetric matrix',
         8:' msolve does not define a pos-def preconditioner'}

    istop  = 0
    w      = 0*b

    # Set up y for the first Lanczos vector v1.
    # y is really beta1 * P * v1  where  P = C^(-1).
    # y and beta1 will be zero if b = 0.

    r1 = 1*b
    y  = B*r1

    beta1 = inner(r1, y)

    # Test for an indefinite preconditioner.
    # If b = 0 exactly, stop with x = 0.

    if beta1 < 0:
        raise ValueError, 'B does not define a pos-def preconditioner'
    if beta1 == 0:
        return r1*0, [0], [], []

    beta1 = sqrt(beta1)
    s     = 1.0 / beta1
    v     = s * y

    y = A*v

    # Set up y for the second Lanczos vector.
    # Again, y is beta * P * v2  where  P = C^(-1).
    # y and beta will be zero or very small if Abar = I or constant * I.

    if shift:
        y -= shift * v
    alfa = inner(v, y)
    y -= (alfa / beta1) * r1

    # Make sure  r2  will be orthogonal to the first  v.

    z  = inner(v, y)
    s  = inner(v, v)
    y -= (z / s) * v
    r2 = y
    y  = B*y

    oldb   = beta1
    beta   = inner(r2, y)
    if beta < 0:
        raise ValueError, 'B does not define a pos-def preconditioner'

    #  Cause termination (later) if beta is essentially zero.

    beta = sqrt(beta)
    if beta <= eps:
        istop = -1

    #  Initialize other quantities.
    rhs2   = 0
    tnorm  = alfa**2 + beta**2
    gbar   = alfa
    dbar   = beta

    bstep  = 0
    ynorm2 = 0
    snprod = 1

    gmin   = gmax   = abs(alfa) + eps
    rhs1   = beta1

    # ------------------------------------------------------------------
    # Main iteration loop.
    # ------------------------------------------------------------------
    # Estimate various norms and test for convergence.

    alphas = []
    betas = []
    residuals = [beta1]
    itn = 0

    while True:
        itn    = itn  +  1
        anorm  = sqrt(tnorm)
        ynorm  = sqrt(ynorm2)
        epsa   = anorm * eps
        epsx   = anorm * ynorm * eps
        epsr   = anorm * ynorm * tolerance
        diag   = gbar

        if diag == 0: diag = epsa

        lqnorm = sqrt(rhs1**2 + rhs2**2)
        qrnorm = snprod * beta1
        cgnorm = qrnorm * beta / abs(diag)

        # Estimate  Cond(A).
        # In this version we look at the diagonals of  L  in the
        # factorization of the tridiagonal matrix,  T = L*Q.
        # Sometimes, T(k) can be misleadingly ill-conditioned when
        # T(k+1) is not, so we must be careful not to overestimate acond

        if lqnorm < cgnorm:
            acond  = gmax / gmin
        else:
            acond  = gmax / min(gmin, abs(diag))

        zbar = rhs1 / diag
        z    = (snprod * zbar + bstep) / beta1

        # See if any of the stopping criteria are satisfied.
        # In rare cases, istop is already -1 from above
        # (Abar = const * I).

        if istop == 0:
            if acond   >= 0.1/eps    : istop = 4
            if epsx    >= beta1      : istop = 3
            if cgnorm  <= epsx       : istop = 2
            if cgnorm  <= epsr       : istop = 1

        residuals.append(cgnorm)

        if istop !=0:
            break

        # Obtain the current Lanczos vector  v = (1 / beta)*y
        # and set up  y  for the next iteration.

        s = 1/beta
        v = s * y
        y = A*v
        if shift:
            y -= shift * v
        y -= (beta / oldb) * r1
        alfa = inner(v, y)
        y -= (alfa / beta) * r2
        r1 = r2.copy()
        r2 = y
        y = B*y
        oldb = beta
        beta = inner(r2, y)

        alphas.append(alfa)
        betas.append(beta)

        if beta < 0:
            raise ValueError, 'A does not define a symmetric matrix'

        beta  = sqrt(beta);
        tnorm = tnorm  +  alfa**2  +  oldb**2  +  beta**2;

        # Compute the next plane rotation for Q.

        gamma  = sqrt(gbar**2 + oldb**2)
        cs     = gbar / gamma
        sn     = oldb / gamma
        delta  = cs * dbar  +  sn * alfa
        gbar   = sn * dbar  -  cs * alfa
        epsln  = sn * beta
        dbar   =            -  cs * beta

        # Update  X.

        z = rhs1 / gamma
        s = z*cs
        t = z*sn
        x += s*w + t*v
        w *= sn
        w -= cs*v

        # Accumulate the step along the direction b, and go round again.

        bstep  = snprod * cs * z  +  bstep
        snprod = snprod * sn
        gmax   = max(gmax, gamma)
        gmin   = min(gmin, gamma)
        ynorm2 = z**2  +  ynorm2
        rhs1   = rhs2  -  delta * z
        rhs2   =       -  epsln * z

    # ------------------------------------------------------------------
    # End of main iteration loop.
    # ------------------------------------------------------------------

    # Move to the CG point if it seems better.
    # In this version of SYMMLQ, the convergence tests involve
    # only cgnorm, so we're unlikely to stop at an LQ point,
    # EXCEPT if the iteration limit interferes.

    if cgnorm < lqnorm:
        zbar   = rhs1 / diag
        bstep  = snprod * zbar + bstep
        ynorm  = sqrt(ynorm2 + zbar**2)
        x     += zbar * w

    # Add the step along b.

    bstep  = bstep / beta1
    y = B*b
    x += bstep * y

    if istop != 1:
        print 'SymmLQ:',msg[istop]

    return x, residuals, [], []

def tfqmr(B, A, x, b, tolerance=1e-5, relativeconv=False, maxiter=200):
    #####
    # Adapted from PyKrylov (https://github.com/dpo/pykrylov; LGPL license)
    #####

    r0 = b - A*x

    rho = inner(r0,r0)
    alphas = []
    betas = []
    residuals = [sqrt(rho)]

    if relativeconv:
        tolerance *= residuals[0]

    if residuals[-1] < tolerance:
        return x, residuals, [], []

    y = r0.copy()   # Initial residual vector
    w = r0.copy()
    d = 0*b
    theta = 0.0
    eta = 0.0
    k = 0

    z = B*y
    u = A*z
    v = u.copy()

    while k < maxiter:

        k += 1
        sigma = inner(r0,v)
        alpha = rho/sigma

        # First pass
        w -= alpha * u
        d *= theta * theta * eta / alpha
        d += z

        residNorm = residuals[-1]
        theta = norm(w)/residNorm
        c = 1.0/sqrt(1 + theta*theta)
        residNorm *= theta * c
        eta = c * c * alpha
        x += eta * d
        m = 2.0 * k - 1.0
        if residNorm * sqrt(m+1) < tolerance:
            break

        # Second pass
        m += 1
        y -= alpha * v
        z = B*y

        u = A*z
        w -= alpha * u
        d *= theta * theta * eta / alpha
        d += z
        theta = norm(w)/residNorm
        c = 1.0/sqrt(1 + theta*theta)
        residNorm *= theta * c
        eta = c * c * alpha
        x += eta * d

        residuals.append(residNorm * sqrt(m+1))
        if residuals[-1] < tolerance or k >= maxiter:
            break

        # Final updates
        rho_next = inner(r0,w)
        beta = rho_next/rho
        rho = rho_next

        alphas.append(alpha)
        betas.append(beta)

        # Update y
        y *= beta
        y += w

        # Partial update of v with current u
        v *= beta
        v += u
        v *= beta

        # Update u
        z = B*y
        u = A*z

        # Complete update of v
        v += u

    return x, residuals, alphas, betas
