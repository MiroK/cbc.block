from __future__ import division

from __future__ import absolute_import
from math import sqrt
import numpy

def inner(x,y):
    return x.inner(y)

def norm(v):
    return v.norm('l2')

def transpmult(A, x):
    return A.transpmult(x)

eps = numpy.finfo(float).eps

