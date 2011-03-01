from __future__ import division
from blockbase import blockbase

class blockcompose(blockbase):
    def __init__(self, A, B):
        """Args may be blockoperators or individual blocks (scalar or Matrix)"""
        A = A.chain if isinstance(A, blockcompose) else [A]
        B = B.chain if isinstance(B, blockcompose) else [B]
        self.chain = B+A
    def __mul__(self, x):
        for op in self.chain:
            x = op * x
            if x == NotImplemented:
                return NotImplemented
        return x
    def __sub__(self, x):
        return blocksub(self, x)
    def __add__(self, x):
        return blockadd(self, x)

class SubBlockOperator(object):
    """Create an operator on a sub-block of a larger block operator. Can be used for example
    to solve the 2x2 Schur complement of a 3x3 block operator"""
    def __init__(self, idx, sub_op):
        self.sub_op = sub_op
        self.idx = idx

    def matvec(self, other):
        from blockoperator import blockvec
        other_sub = blockvec(other.blocks[self.idx])
        result_sub = self.sub_op * other_sub

        result = blockvec(len(other))
        for i in range(len(result_sub)):
            result[self.idx[i]] = result_sub[i]
        for i in range(len(result)):
            if result[i] is None:
                result[i] = other[i]

        return result


# It's probably best if blocksub and blockadd do not allow coercion into
# blockcompose, since that might mess up the operator precedence. Hence, they
# do not inherit from blockbase. As it is now, self.A*x and self.B*x must be
# reduced to vectors, which means all block multiplies are finished before
# __mul__ does anything.

class blocksub(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B
    def __mul__(self, x):
        from blockoperator import blockvec
        from dolfin import Vector
        if not isinstance(x, (Vector, blockvec)):
            return NotImplemented
        y = self.A*x
        y -= self.B*x
        return y
    def __neg__(self):
        return blocksub(self.B, self.A)

class blockadd(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B
    def __mul__(self, x):
        from blockoperator import blockvec
        from dolfin import Vector
        if not isinstance(x, (Vector, blockvec)):
            return NotImplemented
        y = self.A*x
        y += self.B*x
        return y
    def __neg__(self):
        return blockcompose(-1, self)
