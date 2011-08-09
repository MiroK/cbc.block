from __future__ import division

"""Block operations for linear algebra.

To make this work, all operators should define at least a __mul__(self, other)
method, which either does its thing (typically if isinstance(other,
(block_vec, GenericVector))), or returns a block_mul(self, other) object which defers the
action until there is a proper vector to work on.

In addition, methods are injected into dolfin.Matrix / dolfin.Vector as
needed. This should eventually be moved to Dolfin proper.
"""

from block_mat import block_mat
from block_vec import block_vec
from block_compose import block_mul, block_add, block_sub, block_transpose
from block_transform import block_kronecker, block_simplify, block_collapse
from block_bc import block_bc

def _init():
    import dolfin
    from object_pool import vec_pool
    from block_base import block_container

    # To make stuff like L=C*B work when C and B are type dolfin.Matrix, we inject
    # methods into dolfin.Matrix

    def check_type(obj1, obj2):
        if isinstance(obj2, block_container):
            raise TypeError, 'cannot apply dolfin operators on block containers:\n\t%s\nand\n\t%s'%(obj1,obj2)
        return True

    def wrap(cls, name, wrap_to):
        _old_method = getattr(cls, name)
        def _new_method(self, other):
            check_type(self, other)
            y = _old_method(self, other)
            if y == NotImplemented:
                y = wrap_to(self, other)
            return y
        setattr(cls, name, _new_method)

    wrap(dolfin.Matrix, '__mul__', block_mul)
    wrap(dolfin.Matrix, '__add__', block_add)
    wrap(dolfin.Matrix, '__sub__', block_sub)

    dolfin.Matrix.__rmul__ = lambda self, other: check_type(self, other) and block_mul(other, self)
    dolfin.Matrix.__radd__ = lambda self, other: check_type(self, other) and block_add(other, self)
    #dolfin.Matrix.__rsub__ = lambda self, other: check_type(self, other) and block_sub(other, self)
    dolfin.Matrix.__neg__  = lambda self       : block_mul(-1, self)

    # Inject a new transpmult() method that returns the result vector (instead of output parameter)
    old_transpmult = dolfin.Matrix.transpmult
    def transpmult(self, x, y=None):
        check_type(self, x)
        if y is None:
            y = self.create_vec(dim=0)
        old_transpmult(self, x, y)
        return y
    dolfin.Matrix.transpmult = transpmult

    # Inject a create() method that returns the new vector (instead of resize() which uses out parameter)
    def create_vec(self, dim=1):
        vec = dolfin.Vector()
        self.resize(vec, dim)
        return vec
    dolfin.GenericMatrix.create_vec = vec_pool(create_vec)

    # For the Trilinos stuff, it's much nicer if down_cast is a method on the
    # object. Also do some fixup so that the down-casted object is not deleted
    # just because the owning object goes out of scope, by creating a hidden
    # backwards reference. (Python garbage collects circular references as long
    # as custom __del__ methods are not in use.)
    def la_object(self):
        obj = self.la_object()
        obj.reference = self
        return obj
    def down_cast(self):
        obj = dolfin.down_cast(self)
        if not hasattr(obj, 'la_object'):
            cls = obj.__class__
            if hasattr(cls, 'vec'):
                cls.vec, cls.la_object = la_object, cls.vec
            elif hasattr(cls, 'mat'):
                cls.mat, cls.la_object = la_object, cls.mat
            else:
                raise RuntimeError, 'down_cast on unknown object'
        return obj

    dolfin.GenericMatrix.down_cast = down_cast
    dolfin.GenericVector.down_cast = down_cast

_init()
