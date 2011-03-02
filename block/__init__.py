from __future__ import division

"""Block operations for linear algebra.

To make this work, all operators should define at least a __mul__(self, other)
method, which either does its thing (typically if isinstance(other,
BlockVector)), or returns a BlockCompose(self, other) object which defers the
action until there is a BlockVector to work on.

In addition, methods are injected into dolfin.Matrix / dolfin.Vector as needed.

NOTE: Nested blocks SHOULD work but has not been tested.
"""

import dolfin
from blockoperator import blockop
from blockvector import blockvec
from blockcompose import blockcompose
from blockbc import blockbc

# To make stuff like L=C*B work when C and B are type dolfin.Matrix, we inject
# methods into dolfin.Matrix
def _rmul(self, other):
    return blockcompose(other, self)
def _neg(self):
    return blockcompose(-1, self)

_old_mat_mul = dolfin.Matrix.__mul__
def _mat_mul(self, x):
    y = _old_mat_mul(self, x)
    if y == NotImplemented:
        return blockcompose(self, x)
    return y
dolfin.Matrix.__mul__  = _mat_mul

def _mat_rmul(self, other):
    return blockcompose(other, self)
dolfin.Matrix.__rmul__ = _mat_rmul

def _mat_neg(self):
    return blockcompose(-1, self)
dolfin.Matrix.__neg__ = _mat_neg

# For the Trilinos stuff, it's much nicer if down_cast is a method on the object
def _down_cast(self):
    return dolfin.down_cast(self)
dolfin.Matrix.down_cast = _down_cast
dolfin.Vector.down_cast = _down_cast
