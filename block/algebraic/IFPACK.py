from __future__ import division

from dolfin import down_cast, Vector
from block.blockbase import blockbase

class IFPACK(blockbase):

    params = {}

    def __init__(self, A, overlap=0, params={}):
        from PyTrilinos.IFPACK import Factory

        self.A = A # Keep reference to avoid delete

        prectype = self.prectype
        if overlap == 0:
            prectype += ' stand-alone' # Skip the additive Schwarz step

        self.prec = Factory().Create(prectype, down_cast(A).mat(), overlap)
        if not self.prec:
            raise RuntimeError, "Unknown IFPACK preconditioner '%s'"%prectype

        paramlist = {'schwartz: combine mode' : 'Add'} # Slower than 'Zero', but symmetric
        paramlist.update(self.params)
        paramlist.update(params)

        assert (0 == self.prec.SetParameters(paramlist))
        assert (0 == self.prec.Initialize())
        assert (0 == self.prec.Compute())

    def matvec(self, b):
        if not isinstance(b, Vector):
            return NotImplemented
        x = Vector(len(b))
        err = self.prec.ApplyInverse(down_cast(b).vec(), down_cast(x).vec())
        if err:
            raise RuntimeError('ApplyInverse returned %d'%err)
        return x

# "point relaxation" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_PointRelaxation>
# "block relaxation" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_BlockRelaxation>
# "Amesos" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_Amesos>.
# "IC" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_IC>.
# "ICT" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_ICT>.
# "ILU" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_ILU>.
# "ILUT" : returns an instance of Ifpack_AdditiveSchwarz<Ifpack_ILUT>.
# otherwise, Create() returns 0.

class Jacobi(IFPACK):
    prectype = 'point relaxation'
    params = {'relaxation: type' : 'Jacobi'}

class GaussSeidel(IFPACK):
    prectype = 'point relaxation'
    params = {'relaxation: type' : 'Gauss-Seidel'}

class SymmGaussSeidel(IFPACK):
    prectype = 'point relaxation'
    params = {'relaxation: type' : 'symmetric Gauss-Seidel'}

class BJacobi(IFPACK):
    prectype = 'block relaxation'
    params = {'relaxation: type' : 'Jacobi'}

class BGaussSeidel(IFPACK):
    prectype = 'block relaxation'
    params = {'relaxation: type' : 'Gauss-Seidel'}

class BSymmGaussSeidel(IFPACK):
    prectype = 'block relaxation'
    params = {'relaxation: type' : 'symmetric Gauss-Seidel'}

class ILU(IFPACK):
    """Incomplete LU factorization"""
    prectype = 'ILU'

class ILUT(IFPACK):
    """ILU with threshold"""
    prectype = 'ILUT'

class IC(IFPACK):
    """Incomplete Cholesky factorization"""
    prectype = 'IC'

class ICT(IFPACK):
    """IC with threshold"""
    prectype = 'ICT'

class Amesos(IFPACK):
    prectype = 'Amesos'

del IFPACK
