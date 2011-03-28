from __future__ import division

"""Classes that define algebraic operations on matrices by deferring the actual
action until a vector is present. The classes are normally not used directly,
but instead returned implicitly by mathematical notation. For example,

M = A*B+C

returns the object

M = block_add(block_compose(A,B), C),

and the actual calculation is done later, once w=M*v is called for a vector v:

w=block_add.__mul__(M, v)
--> a = block_compose.__mul__((A,B), v)
----> b = Matrix.__mul__(B, v)
----> a = Matrix.__mul__(A, b)
--> b = Matrix.__mul__(C, v)
--> w = a+b
"""

from block_base import block_base

class block_compose(block_base):
    def __init__(self, A, B):
        """Args may be blockoperators or individual blocks (scalar or Matrix)"""
        A = A.chain if isinstance(A, block_compose) else [A]
        B = B.chain if isinstance(B, block_compose) else [B]
        self.chain = A+B
    def __mul__(self, x):
        for op in reversed(self.chain):
            from dolfin import GenericMatrix, GenericVector
            if isinstance(op, GenericMatrix) and isinstance(x, GenericVector):
                y = op.create_vec(dim=0)
                op.mult(x, y)
                x = y
            else:
                x = op * x
            if x == NotImplemented:
                return NotImplemented
        return x

    def transpmult(self, x):
        from numpy import isscalar
        for op in self.chain:
            if isscalar(op):
                if op != 1:
                    x = op*x
            else:
                x = op.transpmult(x)
        return x

    def create_vec(self, dim=1):
        if dim==0:
            for op in self.chain:
                try:
                    return op.create_vec(dim)
                except AttributeError:
                    pass
        if dim==1:
            for op in reversed(self.chain):
                try:
                    return op.create_vec(dim)
                except AttributeError:
                    pass
        raise AttributeError, 'failed to create vec, no appropriate reference matrix'

    def inside_out(self):
        """Create a block_mat of block_composes from a block_compose of block_mats"""
        from block_mat import block_mat
        from numpy import isscalar

        # Reduce all composed objects
        ops = self.chain[:]
        for i,op in enumerate(ops):
            if hasattr(op, 'simplify'):
                ops[i] = ops[i].simplify()
        for i,op in enumerate(ops):
            if hasattr(op, 'inside_out'):
                ops[i] = ops[i].inside_out()

        # Do the fandango
        while len(ops) > 1:
            factor = 1
            B = ops.pop()
            A = ops.pop()

            if isinstance(A, block_mat) and isinstance(B, block_mat):
                m,n = A.blocks.shape
                p,q = B.blocks.shape
                C = block_mat(m,q)
                for row in range(m):
                    for col in range(q):
                        for i in range(n):
                            C[row,col] += A[row,i]*B[i,col]
            elif isinstance(A, block_mat):
                m,n = A.blocks.shape
                C = block_mat(m,n)
                for row in range(m):
                    for col in range(n):
                        C[row,col] = A[row,col]*B
            elif isinstance(B, block_mat):
                m,n = B.blocks.shape
                C = block_mat(m,n)
                for row in range(m):
                    for col in range(n):
                        C[row,col] = A*B[row,col]
            else:
                C = A*B
            ops.append(C)
        return C

    def simplify(self):
        from numpy import isscalar
        operators = []
        scalar = 1.0
        for op in self.chain:
            if hasattr(op, 'simplify'):
                op = op.simplify()
            if isscalar(op):
                scalar *= op
            else:
                operators.append(op)
        if scalar == 0:
            return 0
        if scalar != 1:
            operators.insert(0, scalar)
        ret = block_compose(None, None)
        ret.chain = operators
        return ret

    def __str__(self):
        return '{%s}'%(' * '.join(op.__str__() for op in self.chain))

    def __iter__(self):
        return iter(self.chain)
    def __len__(self):
        return len(self.chain)
    def __getitem__(self, i):
        return self.chain[i]

class block_transpose(block_base):
    def __init__(self, A):
        self.A = A
    def matvec(self, x):
        return self.A.transpmult(x)
    def transpmult(self, x):
        return self.A.__mul__(x)

    def inside_out(self):
        A = self.A.inside_out() if hasattr(self.A, 'inside_out') else self.A
        if not isinstance(A, block_mat):
            return self
        m,n = A.blocks.shape
        ret = block_mat(n,m)
        for i in range(m):
            for j in range(n):
                ret[j,i] = block_transpose(A[i,j])
        return ret

    def simplify(self):
        from numpy import isscalar
        A = self.A.simplify() if hasattr(self.A, 'simplify') else self.A
        if isscalar(A):
            return A
        if isinstance(A, block_transpose):
            return A.A
        return self

    def __str__(self):
        return '<block_transpose of %s>'%str(self.A)
    def __iter__(self):
        return iter([self.A])
    def __len__(self):
        return 1
    def __getitem__(self, i):
        return [self.A][i]

# It's probably best if block_sub and block_add do not allow coercion into
# block_compose, since that might mess up the operator precedence. Hence, they
# do not inherit from block_base. As it is now, self.A*x and self.B*x must be
# reduced to vectors, which means all composed multiplies are finished before
# __mul__ does anything.

class block_sub(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B
    def __mul__(self, x):
        from block_mat import block_vec
        from dolfin import GenericVector
        if not isinstance(x, (GenericVector, block_vec)):
            return NotImplemented
        y = self.A*x
        z = self.B*x
        if len(y) != len(z):
            raise RuntimeError, \
                'incompatible dimensions in matrix subtraction -- %d != %d'%(len(y),len(z))
        y -= z
        return y
    def __neg__(self):
        return block_sub(self.B, self.A)
    def __add__(self, x):
        return block_add(self, x)
    def __sub__(self, x):
        return block_sub(self, x)

    def create_vec(self, dim=1):
        try:
            return self.A.create_vec(dim)
        except AttributeError:
            return self.B.create_vec(dim)

    def simplify(self):
        from numpy import isscalar
        A = self.A.simplify() if hasattr(self.A, 'simplify') else self.A
        B = self.B.simplify() if hasattr(self.B, 'simplify') else self.B
        if isscalar(A) and A==0:
            return -B
        if isscalar(B) and B==0:
            return A
        return A-B

    def inside_out(self, _f=lambda a,b: a-b):
        """Create a block_mat of block_subs from a block_sub of block_mats"""
        from block_mat import block_mat
        from numpy import isscalar

        A = self.A
        B = self.B
        # Reduce all composed objects
        if hasattr(A, 'simplify'):
            A = A.simplify()
        if hasattr(B, 'simplify'):
            B = B.simplify()
        if hasattr(A, 'inside_out'):
            A = A.inside_out()
        if hasattr(B, 'inside_out'):
            B = B.inside_out()

        if isinstance(A, block_mat) and isinstance(B, block_mat):
            m,n = A.blocks.shape
            C = block_mat(m,n)
            for row in range(m):
                for col in range(n):
                    C[row,col] = _f(A[row,col], B[row,col])
        elif isinstance(A, block_mat):
            m,n = A.blocks.shape
            C = block_mat(m,n)
            for row in range(m):
                for col in range(n):
                    C[row,col] = _f(A[row,col], B) if row==col else A[row,col]
        elif isinstance(B, block_mat):
            m,n = B.blocks.shape
            C = block_mat(m,n)
            for row in range(m):
                for col in range(n):
                    C[row,col] = _f(A, B[row,col]) if row==col else B[row,col]
        else:
            C = _f(A, B)
        return C

    def __str__(self):
        return '{%s - %s}'%(self.A.__str__(), self.B.__str__())

    def __iter__(self):
        return iter([self.A, self.B])
    def __len__(self):
        return 2
    def __getitem__(self, i):
        return [self.A, self.B][i]

class block_add(block_sub):
    def __mul__(self, x):
        from block_mat import block_vec
        from dolfin import GenericVector
        if not isinstance(x, (GenericVector, block_vec)):
            return NotImplemented
        y = self.A*x
        z = self.B*x
        if len(y) != len(z):
            raise RuntimeError, \
                'incompatible dimensions in matrix addition -- %d != %d'%(len(y),len(z))
        y += z
        return y

    def simplify(self):
        from numpy import isscalar
        A = self.A.simplify() if hasattr(self.A, 'simplify') else self.A
        B = self.B.simplify() if hasattr(self.B, 'simplify') else self.B
        if isscalar(A) and A==0:
            return B
        if isscalar(B) and B==0:
            return A
        return A+B

    def inside_out(self):
        return block_sub.inside_out(self, _f=lambda a,b: a+b)

    def __neg__(self):
        return block_compose(-1, self)

    def __str__(self):
        return '{%s + %s}'%(self.A.__str__(), self.B.__str__())


def kronecker(A, B):
    """Create the Kronecker (tensor) product of two matrices. The result is
    returned as a product of two block matrices, (A x Ib) * (Ia x B), because
    this will often limit the number of repeated applications of the inner
    operators. However, it also means that A and B must be square since
    otherwise the identities Ia and Ib are not defined.

    To form the Kronecker sum, you can extract (A x Ib) and (Ia x B) like this:
      C,D = kronecker(A,B); sum=C+D
    Similarly, it may be wise to do the inverse separately:
      C,D = kronecker(A,B); inverse = some_invert(D)*ConjGrad(C)
    """

    # A scalar can represent the scaled identity of any dimension, so no need
    # to diagonal-expand it in the following.
    from numpy import isscalar

    if isinstance(B, block_mat) and not isscalar(A):
        n = len(B.blocks)
        C = block_mat.diag(A, n=n)
    else:
        C = A

    if isinstance(A, block_mat) and not isscalar(B):
        m = len(A.blocks)
        if isinstance(B, block_mat):
            D = block_mat(m,m)
            for i in range(m):
                for j in range(m):
                    b = B[i,j]
                    D[i,j] = b if isscalar(b) else block_mat.diag(b, n=m)
        else:
            D = block_mat.diag(B, n=m)
    else:
        D = B

    return C*D
