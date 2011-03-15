from __future__ import division
from common import *

def richardson(B, A, x, b, tolerance, maxiter, progress, relativeconv=False):

    residuals = []

    r = b - A*x
    residuals = [sqrt(inner(r,r))]

    if relativeconv:
        tolerance *= residuals[0]

    iter = 0
    while residuals[-1] > tolerance and iter < maxiter:
        x += B*r
        r = b - A*x

        residuals.append(sqrt(inner(r,r)))
        iter += 1
        progress += 1

    return x, residuals, [], []
