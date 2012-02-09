from block_mat import block_mat
from block_compose import block_mul

def block_kronecker(A, B):
    """Create the Kronecker (tensor) product of two matrices. The result is
    returned as a product of two block matrices, (A x Ib) * (Ia x B), because
    this will often limit the number of repeated applications of the inner
    operators. However, it also means that A and B must be square since
    otherwise the identities Ia and Ib are not defined.

    Note: If none of the parameters are of type block_mat, the composition A*B
    is returned instead of two block matrices.

    To form the Kronecker sum, you can extract (A x Ib) and (Ia x B) like this:
      C,D = block_kronecker(A,B); sum=C+D
    Similarly, it may be wise to do the inverse separately:
      C,D = block_kronecker(A,B); inverse = some_invert(D)*ConjGrad(C)
    """
    from numpy import isscalar

    if isinstance(B, block_mat):
        n = len(B.blocks)
        C = block_mat.diag(A, n=n)
    else:
        C = A

    if isinstance(A, block_mat):
        m = len(A.blocks)
        if isinstance(B, block_mat):
            D = block_mat(m,m)
            for i in range(m):
                for j in range(m):
                    # A scalar can represent the scaled identity of any
                    # dimension, so no need to diagonal-expand it here. We
                    # don't do this check on the outer diagonal expansions,
                    # because it is clearer to always return two block matrices
                    # of equal dimension rather than sometimes a scalar.
                    b = B[i,j]
                    D[i,j] = b if isscalar(b) else block_mat.diag(b, n=m)
        else:
            D = block_mat.diag(B, n=m)
    else:
        D = B

    return block_mul(C,D)


def block_simplify(expr):
    """Return a simplified (if possible) representation of a block matrix or
    block composition. The simplification does the following basic steps:
    - Convert scaled identity matrices to scalars
    - Combine scalar terms in compositions (2*2==4)
    - Eliminate additive and multiplicative identities (A+0=A, A*1=A)
    """
    if hasattr(expr, 'block_simplify'):
        return expr.block_simplify()
    else:
        return expr


def block_collapse(expr):
    """Turn a composition /inside out/, i.e., turn a composition of block
    matrices into a block matrix of compositions.
    """
    if hasattr(expr, 'block_collapse'):
        return expr.block_collapse()
    else:
        return expr
