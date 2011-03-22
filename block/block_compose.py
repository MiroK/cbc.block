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
        self.chain = B+A
    def __mul__(self, x):
        for op in self.chain:
            from dolfin import GenericMatrix
            if isinstance(op, GenericMatrix):
                y = op.create_vec(dim=0)
                op.mult(x, y)
                x = y
            else:
                x = op * x
            if x == NotImplemented:
                return NotImplemented
        return x
    def __sub__(self, x):
        return block_sub(self, x)
    def __add__(self, x):
        return block_add(self, x)
    def __radd__(self, x):
        return block_add(x, self)

    def transpmult(self, x):
        from numpy import isscalar
        for op in reversed(self.chain):
            if isscalar(op):
                if op != 1:
                    x = op*x
            else:
                x = op.transpmult(x)
        return x

    def create_vec(self, dim=1):
        # dim is 0 or 1, use first or last operator in chain
        return self.chain[dim-1].create_vec(dim)

    def __str__(self):
        return '{%s}'%(' * '.join(op.__str__() for op in reversed(self.chain)))


class block_transpose(block_base):
    def __init__(self, A):
        self.A = A
    def matvec(self, x):
        return self.A.transpmult(x)
    def transpmult(self, x):
        return self.A.__mul__(x)

    def __str__(self):
        return '<block_transpose of %s>'%str(self.A)

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

    def create_vec(self, dim=1):
        return self.A.create_vec(dim)

    def __str__(self):
        return '{%s - %s}'%(self.A.__str__(), self.B.__str__())

class block_add(object):
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
                'incompatible dimensions in matrix addition -- %d != %d'%(len(y),len(z))
        y += z
        return y
    def __neg__(self):
        return block_compose(-1, self)

    def create_vec(self, dim=1):
        return self.A.create_vec(dim)

    def __str__(self):
        return '{%s + %s}'%(self.A.__str__(), self.B.__str__())

