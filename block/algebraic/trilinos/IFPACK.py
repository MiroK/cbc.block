from __future__ import division

from block.block_base import block_base

class IFPACK(block_base):

    errcode = {1 : "Generic Error (called method or function returned an error)",
               2 : "Input data not valid (wrong parameter, out-of-bounds, wrong dimensions, matrix is not square,...)",
               3 : "Data has not been correctly pre-processed",
               4 : "Problem encountered during application of the algorithm (division by zero, out-of-bounds, ...)",
               5 : "Memory allocation error",
               98: "Feature is not supported",
               99: "Feature is not implemented yet (check Known Bugs and Future Developments, or submit a bug)"}

    params = {}

    def __init__(self, A, overlap=0, params={}):
        from PyTrilinos.IFPACK import Factory

        self.A = A # Keep reference to avoid delete

        prectype = self.prectype
        if overlap == 0:
            prectype += ' stand-alone' # Skip the additive Schwarz step

        self.prec = Factory().Create(prectype, A.down_cast().mat(), overlap)
        if not self.prec:
            raise RuntimeError, "Unknown IFPACK preconditioner '%s'"%prectype

        paramlist = {'schwartz: combine mode' : 'Add'} # Slower than 'Zero', but symmetric
        paramlist.update(self.params)
        paramlist.update(params)

        assert (0 == self.prec.SetParameters(paramlist))
        assert (0 == self.prec.Initialize())
        assert (0 == self.prec.Compute())

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented
        x = self.A.create_vec()
        err = self.prec.ApplyInverse(b.down_cast().vec(), x.down_cast().vec())
        if err:
            raise RuntimeError('ApplyInverse returned error %d: %s'%(err, self.errcode.get(-err)))
        return x

    def down_cast(self):
        return self.prec

    def __str__(self):
        return '<%s prec of %s>'%(self.__class__.__name__, str(self.A))

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
