class BlockOperator(BlockThingy):
    """Block of matrices or other operators. Empty blocks doesn't need to be set
    (they may be None or zero), but each row must have at least one non-empty block.

    As a special case, the diagonal may contain scalars which act as the
    (scaled) identity operators."""

    def __init__(self, m, n=None):
        if n is None:
            BlockThingy.__init__(self, blocks=m)
        else:
            BlockThingy.__init__(self, mn=(m,n))

    def matvec(self, x):
        m,n = self.blocks.shape
        y = BlockVector(m)

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
        # Probably incorrect, since BiCGStab and CGN both fail...
        m,n = self.blocks.shape
        y = BlockVector(len(r))

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
        y = BlockOperator(m,n)
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

    def schur_approximation(self, symmetry=1.0, scale=1):
        """Create an explicit approximation of the Schur complement. The symmetry can be
        declared, +1 for symmetric and -1 for antisymmetric blocks. Setting scale=-1 may be
        handy, because it will usually return a PD matrix (negative Schur complement)."""
        if self.blocks.shape != (2,2):
            raise ValueError, "must be 2x2 blocks"
        bm = BlockMatrix(2,2)
        bm[0,0] = self[0,0]
        bm[0,1] = self[0,1]
        bm[1,0] = self[1,0]
        bm[1,1] = self[1,1]
        S = bm.schur_approximation(symmetry)
        if scale != 1:
            S *= scale
        return S

    def scheme(self, name, reverse=False):
        return BlockScheme(self, name, reverse)

