from __future__ import division
import numpy
import copy
from dolfin import Matrix, Vector, BlockMatrix, info, warning, error

"""Block operations for linear algebra.

To make this work, all operators should define at least a __mul__(self, other)
method, which either does its thing (typically if isinstance(other,
BlockVector)), or returns a BlockCompose(self, other) object which defers the
action until there is a BlockVector to work on.

In addition, methods are injected into dolfin.Matrix / dolfin.Vector as needed.

NOTE: Nested blocks SHOULD work but has not been tested.
"""

# To make stuff like L=C*B work when C and B are type dolfin.Matrix, we inject
# methods into dolfin.Matrix
def _rmul(self, other):
    return BlockCompose(other, self)
def _neg(self):
    return BlockCompose(-1, self)

_old_mat_mul = Matrix.__mul__
def _mat_mul(self, x):
    y = _old_mat_mul(self, x)
    if y == NotImplemented:
        return BlockCompose(self, x)
    return y
Matrix.__mul__  = _mat_mul

def _mat_rmul(self, other):
    return BlockCompose(other, self)
Matrix.__rmul__ = _mat_rmul

def _mat_neg(self):
    return BlockCompose(-1, self)
Matrix.__neg__ = _mat_neg
