from __future__ import division
from block_base import block_container

class block_vec(block_container):
    """Class defining a block vector suitable for multiplication with a
    block_mat of the right dimension. Many of the methods of dolfin.Vector are
    made available by calling the equivalent method on the individual
    vectors."""

    def __init__(self, m, blocks=None):
        if hasattr(m, '__iter__'):
            blocks = m
            m = len(m)
        block_container.__init__(self, m, blocks)

    def allocate(self, AA, dim=1):
        from dolfin import GenericVector
        for i in range(len(self)):
            if isinstance(self[i], GenericVector):
                continue
            for j in range(len(self)):
                A = AA[i,j] if dim==0 else AA[j,i]
                if hasattr(A, 'create_vec'):
                    self[i] = A.create_vec(dim)
                    break
            if not isinstance(self[i], GenericVector):
                raise RuntimeError, "can't allocate vector - no Matrix (or equivalent) for block %d"%i

    def norm(self, ntype='l2'):
        if ntype == 'linf':
            return max(x.norm(ntype) for x in self)
        else:
            try:
                assert(ntype[0] == 'l')
                p = int(ntype[1:])
            except:
                raise TypeError("Unknown norm '%s'"%ntype)
            unpack = lambda x: pow(x, p)
            pack   = lambda x: pow(x, 1/p)
            return pack(sum(unpack(x.norm(ntype)) for x in self))

    def randomize(self):
        import numpy
        for i in range(len(self)):
            if numpy.isscalar(self[i]):
                raise RuntimeError, \
                    'block %d in block_vec has no size -- use proper vector or call allocate(A)' % i
            ran = numpy.random.random(self[i].local_size())
            self[i].set_local(ran)

    def _map_operator(self, operator):
        y = block_vec(len(self))
        for i in range(len(self)):
            y[i] = getattr(self[i], operator)()
        return y

    def _map_scalar_operator(self, operator, x, y=None):
        try:
            x = float(x)
        except:
            return NotImplemented
        if y is None:
            y = block_vec(len(self))
        for i in range(len(self)):
            y[i] = getattr(self[i], operator)(x)
            if y[i] == NotImplemented: return NotImplemented
        return y

    def _map_vector_operator(self, operator, x, y=None):
        if y is None:
            y = block_vec(len(self))
        for i in range(len(self)):
            y[i] = getattr(self[i], operator)(x[i])
            if y[i] == NotImplemented: return NotImplemented
        return y


    def copy(self): return self._map_operator('copy')
    def zero(self): return self._map_operator('zero')

    def __add__ (self, x): return self._map_vector_operator('__add__',  x)
    def __radd__(self, x): return self._map_vector_operator('__radd__', x)
    def __iadd__(self, x): return self._map_vector_operator('__iadd__', x, self)

    def __sub__ (self, x): return self._map_vector_operator('__sub__',  x)
    def __rsub__(self, x): return self._map_vector_operator('__rsub__', x)
    def __isub__(self, x): return self._map_vector_operator('__isub__', x, self)

    def __mul__ (self, x): return self._map_scalar_operator('__mul__',  x)
    def __rmul__(self, x): return self._map_scalar_operator('__rmul__', x)
    def __imul__(self, x): return self._map_scalar_operator('__imul__', x, self)

    def inner(self, x):
        y = self._map_vector_operator('inner', x)
        if y == NotImplemented:
            raise NotImplementedError('One or more blocks do not implement .inner()')
        return sum(y)
