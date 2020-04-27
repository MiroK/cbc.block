from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from .common import *
import numpy as np
from six.moves import zip


def subminres(B, A, x, b, tolerance, maxiter, progress, relativeconv=False, shift=0, callback=None):
    '''
    This code implements the modified version of MINRES with residual
    subvector norms as described in

     Herzog, Soodhalter: A modified implementation of MINRES to monitor 
     residual subvector norms for block systems (submitted)

    Adapted from MARLAB code by Roland Herzog and released under CC-BY-SA 4.0.
    If this code is used in a scientific publication, please cite as
 
     Roland Herzog, Kirk M. Soodhalter: 
     SUBMINRES. A modified implementation of  MINRES to monitor 
     residual subvector norms for block systems,
     Version 1.0 (20160310)
     DOI 10.5281/zenodo.TODO
    '''
    # Initialize some vectors and scalar quantities (lines 1-2)
    # Initialize iterates of v,w,x and initial residual
    vjm1 = x.copy(); vjm1.zero()
    rj = b - A*x
    vj = rj.copy()
    wjm1 = vjm1.copy()  # 0
    wj = wjm1.copy()    # 0

    zj = B*vj
    
    # Normalize the quantities v and z (line 3)
    gamma2j = vj.inner(zj)
    # Check action of preconditioner
    if gamma2j < 0:
        raise ValueError('Preconditioner was not symmetric positive definite')

    # Evaluate the P-norm of initial residual 
    gammaj = np.sqrt(gamma2j)
    zj *= 1./gammaj
    vj *= 1./gammaj

    # Compute initial partial duality products (line 4)
    psij = np.array([zjk.inner(vjk) for zjk, vjk in zip(zj, vj)])

    # Check action of preconditioner
    if any(psijk < 0 for psijk in psij):
        raise ValueError('Preconditioner was not symmetric positive definite')

    # Perform further initializations (lines 5-7)
    # Initialize squared residual P-norm fractions and m vector
    mujm1 = 1.0*psij
    mj = vj.copy()

    # Assign vector of P-norms of subvectors and total initial residual
    etajm1 = np.r_[np.sqrt(psij) * gammaj, gammaj] 
    eta0 = 1.0*etajm1

    # Initialize sines and cosines
    sjm1 = 0
    sj = 0
    cjm1 = 1
    cj = 1

    # Convergence is achieved when the P-norms of the total residual vector
    # and all of its subvectors verify both their relative and absolute tolerances
    if relativeconv:
        # NOTE: Sometimes (e.g. x == 0) can lead to some of the elements of
        # elm0 to be zero. In this case we substite for zeros the mean of
        # the error
        eta0_cvrg = np.abs(eta0)
        eta0_cvrg[eta0_cvrg < 1E-15] = np.mean(eta0_cvrg)
        
        small_enough = lambda array: all(abs(elm) < tolerance*elm0
                                         for elm, elm0 in zip(array, eta0_cvrg))
    else:
        small_enough = lambda array: all(abs(elm) < tolerance for elm in array)

    # Call is with current residuals and solution. Has converged?
    if callback is None:
        # NOTE bool(None) is False
        converged_by_cb = lambda niter, norms, solution: False
    else:
        converged_by_cb = callback

    niter = 1
    residuals = [np.abs(etajm1)]
    # Main loop (lines 8-29) 
    while not small_enough(etajm1) and niter <= maxiter:
        # Main body 
        # Evaluate matrix times vector and update delta and v (lines 9-10)
        Azj = A*zj
        deltaj = zj.inner(Azj)
        vjp1 = Azj - deltaj*vj - gammaj*vjm1

        # Apply the preconditioner (line 11)
        zjp1 = B*vjp1
        # Evaluate duality product (line 12)
        gamma2jp1 = vjp1.inner(zjp1)
        # Check action of preconditioner
        if gamma2jp1 < 0:
            raise ValueError('Preconditioner was not symmetric positive definite')
        
        gammajp1 = np.sqrt(gamma2jp1);
        
        # Normalize the quantities v and z (line 13-14)
        zjp1 *= 1./gammajp1;
        vjp1 *= 1./gammajp1;

        # Update QR factorization (line 15-19)
        alpha0 = cj*deltaj - cjm1*sj*gammaj
        alpha1 = np.sqrt(alpha0**2 + gammajp1**2)
        alpha2 = sj*deltaj + cjm1*cj*gammaj
        alpha3 = sjm1*gammaj
        cjp1 = alpha0/alpha1
        sjp1 = gammajp1/alpha1

        # Update partial duality products psi and theta (lines 20-21)
        thetajp1 = np.array([mjk.inner(zjp1k) for mjk, zjp1k in zip(mj, zjp1)])
        psijp1 = np.array([zjp1k.inner(vjp1k) for zjp1k, vjp1k in zip(zjp1, vjp1)])

        # Update m,w,x vectors (lines 22-24)
        mjp1 = -sjp1*mj + cjp1*vjp1
        wjp1 = (zj - alpha3*wjm1 - alpha2*wj)/alpha1
        x += cjp1*etajm1[-1]*wjp1

        # Update squared residual P-norm fractions (lines 25-26)
        muj = sjp1**2*mujm1 - 2*sjp1*cjp1*thetajp1 + cjp1**2*psijp1

        # Update total and partial residual P-norms (lines 27-28)
        etaj  = -sjp1*etajm1[-1]
        etaj = np.r_[np.sqrt(muj)*etaj, etaj]

        residuals.append(np.abs(etaj))
        print(niter, residuals[-1])

        if converged_by_cb(niter, residuals[-1], x): break

        # Prepare iterates for the next round
        vjm1 = vj; vj = vjp1
        wjm1 = wj; wj = wjp1 
        etajm1 = etaj
        mj = mjp1
        mujm1 = muj
        psij = psijp1
        zj = zjp1
        gammaj = gammajp1
        cjm1 = cj; cj = cjp1
        sjm1 = sj; sj = sjp1

        # Check for maximum # of iterations
        niter += 1
    # Output just the last (global/not-componentwise) residial
    sub_residuals = np.array(residuals)
    
    return x, sub_residuals[:, -1], [], []
