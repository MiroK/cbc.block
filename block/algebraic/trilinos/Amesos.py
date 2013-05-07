from __future__ import division

from block.block_base import block_base

class AmesosSolver(block_base):
    """Trilinos interface to direct solvers. The available solvers depend on
    your Trilinos installation, but the default (Klu) is usually available.

    Serial solvers: Klu, Lapack, Umfpack, Taucs, Superlu

    Parallelsolvers: Superludist, Mumps, Dscpack, Pardiso, Paraklete
    """
    def __init__(self, A, solver='Klu'):
        from PyTrilinos import Epetra, Amesos
        from dolfin import info
        from time import time
        self.A = A # Keep reference
        #self.b = A.create_vec(dim=0)
        #self.x = A.create_vec(dim=1)
        #problem = Epetra.LinearProblem(A.down_cast().mat(),
        #                               self.x.down_cast().vec(),
        #                               self.b.down_cast().vec())
        T = time()
        self.problem = Epetra.LinearProblem()
        self.problem.SetOperator(A.down_cast().mat())
        self.solver = Amesos.Factory().Create(solver, self.problem)
        if self.solver is None:
            raise RuntimeError("Unknown solver '%s'"%solver)
        err = self.solver.SymbolicFactorization()
        if err != 0:
            raise RuntimeError("Amesos " + solver + " symbolic factorization failed, err=%d"%err)
        err = self.solver.NumericFactorization()
        if err != 0:
            raise RuntimeError("Amesos " + solver + " numeric factorization failed, err=%d"%err)
        info('constructed direct solver (using %s) in %.2f s'%(solver,time()-T))

    @staticmethod
    def query(which=None):
        """Return list of available solver backends, or True/False if given a solver name"""
        from PyTrilinos import Amesos
        factory = Amesos.Factory()
        if which is None:
            avail = []
            for s in ['Klu', 'Lapack', 'Umfpack', 'Taucs', 'Superlu',
                      'Superludist', 'Mumps', 'Taucs', 'Superlu']:
                if factory.Query(s):
                    avail.append(s)
            return avail
        return factory.Query(which)

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented()
        if self.A.size(0) != len(b):
            raise RuntimeError(
                'incompatible dimensions for Amesos matvec, %d != %d'%(len(self.b),len(b)))
        x = self.A.create_vec(dim=1)

        self.problem.SetLHS(x.down_cast().vec())
        self.problem.SetRHS(b.down_cast().vec())
        err = self.solver.Solve()
        self.problem.SetLHS(None)
        self.problem.SetRHS(None)

        if err != 0:
            raise RuntimeError("Amesos solve failed, err=%d"%err)
        return x

    def __str__(self):
        return '<%s solver for %s>'%(self.__class__.__name__, str(self.A))
