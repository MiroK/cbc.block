from __future__ import division

from math import sqrt
import numpy

def inner(x,y):
    return x.inner(y)

def norm(v):
    return v.norm('l2')

eps = numpy.finfo(float).eps

