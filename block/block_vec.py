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

    def allocate(self, AA, dim=0):
        """Make sure all blocks are proper vectors. Any non-vector blocks are
        replaced with appropriately sized vectors (where the sizes are taken
        from AA, which should be a block_mat). If dim==0, newly allocated
        vectors use layout appropriate for b (in Ax=b); if dim==1, the layout
        for x is used."""
        from dolfin import GenericVector
        for i in range(len(self)):
            if isinstance(self[i], GenericVector):
                continue
            for j in range(len(self)):
                A = AA[i,j] if dim==0 else AA[j,i]
                try:
                    self[i] = A.create_vec(dim)
                    break
                except AttributeError:
                    pass
            if not isinstance(self[i], GenericVector):
                raise RuntimeError("can't allocate vector - no Matrix (or equivalent) for block %d"%i)

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
        """Fill the block_vec with random data (with zero bias)."""
        import numpy
        for i in range(len(self)):
            if hasattr(self[i], 'local_size'):
                ran = numpy.random.random(self[i].local_size())
                ran -= sum(ran)/len(ran)
                self[i].set_local(ran)
            elif hasattr(self[i], '__len__'):
                ran = numpy.random.random(len(self[i]))
                ran -= sum(ran)/len(ran)
                self[i][:] = ran
            else:
                raise RuntimeError(
                    'block %d in block_vec has no size -- use a proper vector or call allocate(A)' % i)

    #
    # Map operator on the block_vec to operators on the individual blocks.
    #

    def _map_operator(self, operator, inplace=False):
        y = block_vec(len(self))
        for i in range(len(self)):
            try:
                y[i] = getattr(self[i], operator)()
            except Exception, e:
                if i==0 or not inplace:
                    raise e
                else:
                    raise RuntimeError(
                        "operator partially applied, block %d does not support '%s' (err=%s)" % (i, operator, str(e)))
        return y

    def _map_scalar_operator(self, operator, x, inplace=False):
        try:
            x = float(x)
        except:
            return NotImplemented
        y = self if inplace else block_vec(len(self))
        for i in range(len(self)):
            v = getattr(self[i], operator)(x)
            if v == NotImplemented:
                if i==0 or not inplace:
                    return NotImplemented
                else:
                    raise RuntimeError(
                        "operator partially applied, block %d does not support '%s'" % (i, operator))
            y[i] = v
        return y

    def _map_vector_operator(self, operator, x, inplace=False):
        y = self if inplace else block_vec(len(self))
        for i in range(len(self)):
            v = getattr(self[i], operator)(x[i])
            if v == NotImplemented:
                if i==0 or not inplace:
                    return NotImplemented
                else:
                    raise RuntimeError(
                        "operator partially applied, block %d does not support '%s'" % (i, operator))
            y[i] = v
        return y


    def copy(self): return self._map_operator('copy')
    def zero(self): return self._map_operator('zero', True)

    def __add__ (self, x): return self._map_vector_operator('__add__',  x)
    def __radd__(self, x): return self._map_vector_operator('__radd__', x)
    def __iadd__(self, x): return self._map_vector_operator('__iadd__', x, True)

    def __sub__ (self, x): return self._map_vector_operator('__sub__',  x)
    def __rsub__(self, x): return self._map_vector_operator('__rsub__', x)
    def __isub__(self, x): return self._map_vector_operator('__isub__', x, True)

    def __mul__ (self, x): return self._map_scalar_operator('__mul__',  x)
    def __rmul__(self, x): return self._map_scalar_operator('__rmul__', x)
    def __imul__(self, x): return self._map_scalar_operator('__imul__', x, True)

    def inner(self, x):
        y = self._map_vector_operator('inner', x)
        if y == NotImplemented:
            raise NotImplementedError('One or more blocks do not implement .inner()')
        return sum(y)
