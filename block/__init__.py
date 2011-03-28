from __future__ import division

"""Block operations for linear algebra.

To make this work, all operators should define at least a __mul__(self, other)
method, which either does its thing (typically if isinstance(other,
(block_vec, GenericVector))), or returns a block_compose(self, other) object which defers the
action until there is a proper vector to work on.

In addition, methods are injected into dolfin.Matrix / dolfin.Vector as
needed. This should eventually be moved to Dolfin proper.
"""

import dolfin
from block_mat import block_mat
from block_vec import block_vec
from block_compose import block_compose, block_add, block_sub, block_transpose
from block_transform import block_kronecker, block_simplify, block_collapse
from block_bc import block_bc

# To make stuff like L=C*B work when C and B are type dolfin.Matrix, we inject
# methods into dolfin.Matrix

def _wrap(cls, name, wrap_to):
    _old_method = getattr(cls, name)
    def _new_method(self, other):
        y = _old_method(self, other)
        if y == NotImplemented:
            y = wrap_to(self, other)
        return y
    setattr(cls, name, _new_method)

_wrap(dolfin.Matrix, '__mul__', block_compose)
_wrap(dolfin.Matrix, '__add__', block_add)
_wrap(dolfin.Matrix, '__sub__', block_sub)
del _wrap

dolfin.Matrix.__rmul__ = lambda self, other: block_compose(other, self)
dolfin.Matrix.__radd__ = lambda self, other: block_add(other, self)
#dolfin.Matrix.__rsub__ = lambda self, other: block_sub(other, self)
dolfin.Matrix.__neg__  = lambda self       : block_compose(-1, self)

# Inject a new transpmult() method that returns the result vector (instead of output parameter)
def _wrap_transpmult():
    _old_transpmult = dolfin.Matrix.transpmult
    def _transpmult(self, x):
        y = dolfin.Vector()
        _old_transpmult(self, x, y)
        return y
    dolfin.Matrix.transpmult = _transpmult
_wrap_transpmult()
del _wrap_transpmult

# Inject a create() method that returns the new vector (instead of resize() which uses out parameter)
def _create_vec(self, dim=1):
    vec = dolfin.Vector()
    self.resize(vec, dim)
    return vec
from object_pool import vec_pool
dolfin.GenericMatrix.create_vec = vec_pool(_create_vec)
del _create_vec, vec_pool

# For the Trilinos stuff, it's much nicer if down_cast is a method on the object
dolfin.GenericMatrix.down_cast = dolfin.down_cast
dolfin.GenericVector.down_cast = dolfin.down_cast
