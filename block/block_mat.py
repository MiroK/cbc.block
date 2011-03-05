from __future__ import division
from block_base import block_container
from block_vec import block_vec

class block_mat(block_container):
    """Block of matrices or other operators. Empty blocks doesn't need to be set
    (they may be None or zero), but each row must have at least one non-empty block.

    As a special case, the diagonal may contain scalars which act as the
    (scaled) identity operators."""

    def __init__(self, m, n=None):
        if n is None:
            block_container.__init__(self, blocks=m)
        else:
            block_container.__init__(self, mn=(m,n))

    def matvec(self, x):
        m,n = self.blocks.shape
        y = block_vec(m)

        for i in range(m):
            for j in range(n):
                if self[i,j] is None or self[i,j]==0:
                    # Skip multiply if zero
                    continue
                if self[i,j] == 1:
                    # Skip multiply if identity
                    z = x[j]
                else:
                    # Do the block multiply
                    z = self[i,j] * x[j]
                    if z == NotImplemented: return NotImplemented
                if y[i] is None:
                    y[i]  = z
                else:
                    y[i] += z
        return y

    def transpmult(self, x, r):
        import numpy
        from dolfin import Vector
        m,n = self.blocks.shape
        y = block_vec(len(r))

        for i in range(n):
            for j in range(m):
                if self[j,i] is None or self[j,i]==0:
                    # Skip multiply if zero
                    continue
                if self[i,j] == 1:
                    # Skip multiply if identity
                    z = x[j]
                elif numpy.isscalar(self[j,i]):
                    # mult==transpmult
                    z = self[j,i]*x[j]
                else:
                    # Do the block multiply
                    z = Vector()
                    self[j,i].transpmult(x[j], z)
                if y[i] is None:
                    y[i] = z
                else:
                    y[i] += z
        for i in range(m):
            r[i] = y[i]

    def copy(self):
        import copy
        m,n = self.blocks.shape
        y = block_mat(m,n)
        for i in range(m):
            for j in range(n):
                obj = self[i,j]
                if hasattr(obj, 'copy'):
                    y[i,j] = obj.copy()
                else:
                    try:
                        y[i,j] = copy.deepcopy(obj)
                    except TypeError:
                        warning("Don't know how to make a deep copy of (%d,%d), making shallow copy"%(i,j))
                        y[i,j] = copy.copy(obj)
        return y

    def create_vec(self):
        m,n = self.blocks.shape
        xx = block_vec(n)
        xx[:] = 0
        xx.allocate(self)
        return xx

    def scheme(self, name, reverse=False):
        from block_scheme import blockscheme
        return blockscheme(self, name, reverse)