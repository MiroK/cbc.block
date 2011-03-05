from __future__ import division
from block_base import block_base

class block_compose(block_base):
    def __init__(self, A, B):
        """Args may be blockoperators or individual blocks (scalar or Matrix)"""
        A = A.chain if isinstance(A, block_compose) else [A]
        B = B.chain if isinstance(B, block_compose) else [B]
        self.chain = B+A
    def __mul__(self, x):
        for op in self.chain:
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

# It's probably best if block_sub and block_add do not allow coercion into
# block_compose, since that might mess up the operator precedence. Hence, they
# do not inherit from block_base. As it is now, self.A*x and self.B*x must be
# reduced to vectors, which means all block multiplies are finished before
# __mul__ does anything.

class block_sub(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B
    def __mul__(self, x):
        from block_mat import block_vec
        from dolfin import Vector
        if not isinstance(x, (Vector, block_vec)):
            return NotImplemented
        y = self.A*x
        y -= self.B*x
        return y
    def __neg__(self):
        return block_sub(self.B, self.A)

class block_add(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B
    def __mul__(self, x):
        from block_mat import block_vec
        from dolfin import Vector
        if not isinstance(x, (Vector, block_vec)):
            return NotImplemented
        y = self.A*x
        y += self.B*x
        return y
    def __neg__(self):
        return block_compose(-1, self)
