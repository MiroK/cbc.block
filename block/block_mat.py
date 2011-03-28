from __future__ import division
from block_base import block_container
from block_vec import block_vec

class block_mat(block_container):
    """Block of matrices or other operators. Empty blocks doesn't need to be set
    (they may be None or zero), but each row must have at least one non-empty block.

    As a special case, the diagonal may contain scalars which act as the
    (scaled) identity operators."""

    def __init__(self, m, n=None, blocks=0):
        if n is None:
            blocks = m
            m = len(blocks)
            n = len(blocks[0]) if m else 0
        block_container.__init__(self, (m,n), blocks)

    def matvec(self, x):
        from dolfin import GenericVector, GenericMatrix
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
                    if isinstance(self[i,j], GenericMatrix):
                        z = self[i,j].create_vec(dim=0)
                        self[i,j].mult(x[j], z)
                    else:
                        z = self[i,j] * x[j]
                        if z == NotImplemented: return NotImplemented
                if not isinstance(z, (GenericVector, block_vec)):
                    # Typically, this happens when for example a
                    # block_vec([0,0]) is used without calling allocate() or
                    # setting BCs. The result is a Matrix*scalar=Matrix. One
                    # way to fix this issue is to convert all scalars to a
                    # proxy class in block_vec.__init__, and let this proxy
                    # class have a __rmul__ that converts to vector on demand.
                    # (must also stop conversion anything->blockcompose for
                    # this case)
                    raise RuntimeError, \
                        'unexpected result in matvec, %s\n-- possibly because RHS contains scalars ' \
                        'instead of vectors, use create_vec() or allocate()' % type(z)
                if y[i] is None:
                    y[i]  = z
                else:
                    if len(y[i]) != len(z):
                        raise RuntimeError, \
                            'incompatible dimensions in block (%d,%d) -- %d, was %d'%(i,j,len(z),len(y[i]))
                    y[i] += z
        return y

    def transpmult(self, x):
        import numpy
        from dolfin import GenericVector

        m,n = self.blocks.shape
        y = block_vec(self.blocks.shape[0])

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
                    z = self[j,i].transpmult(x[j])
                if not isinstance(z, (GenericVector, block_vec)):
                    # see comment in matvec
                    raise RuntimeError, \
                        'unexpected result in matvec, %s\n-- possibly because RHS contains scalars ' \
                        'instead of vectors, use create_vec() or allocate()' % type(z)
                if y[i] is None:
                    y[i] = z
                else:
                    if len(y[i]) != len(z):
                        raise RuntimeError, \
                            'incompatible dimensions in block (%d,%d) -- %d, was %d'%(i,j,len(z),len(y[i]))
                    y[i] += z
        return y

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

    def create_vec(self, dim=1):
        xx = block_vec(self.blocks.shape[dim])
        xx[:] = 0
        xx.allocate(self, dim)
        return xx

    def scheme(self, name, **kwargs):
        from block_scheme import blockscheme
        return blockscheme(self, name, **kwargs)

    @staticmethod
    def diag(A, n=0):
        """Create a diagonal block matrix, where the entries on the diagonal
        are either the entries of the vector A (if n==0), or n copies of A (if
        n>0). For the case of extracting the diagonal of an existing block
        matrix, use D=A.scheme('jacobi') instead.
        """
        if n==0:
            n = len(A)
            mat = block_mat(n,n)
            for i in range(n):
                mat[i,i] = A[i]
        else:
            mat = block_mat(n,n)
            for i in range(n):
                mat[i,i] = A
        return mat

    def simplify(self):
        """Try to convert identities to scalars, recursively. A fuller
        explanation is found in block_transform.simplify.
        """
        from numpy import isscalar
        from block_transform import simplify
        m,n = self.blocks.shape
        res = block_mat(m,n)
        # Recursive call
        for i in range(m):
            for j in range(n):
                res[i,j] = simplify(self[i,j])
        # Check if result after recursive conversion is the (scaled) identity
        v0 = res.blocks[0,0]
        if m != n:
            return res
        for i in range(m):
            for j in range(n):
                block = res.blocks[i,j]
                if not isscalar(block):
                    return res
                if i==j:
                    if block!=v0:
                        return res
                else:
                    if block!=0:
                        return res
        return v0
