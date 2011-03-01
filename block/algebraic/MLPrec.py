from __future__ import division

import sys
if 'dolfin' in sys.modules and not 'PyTrilinos' in sys.modules:
    raise RuntimeError('must be imported before dolfin -- add "import PyTrilinos" first in your script')

from PyTrilinos import ML
from PyTrilinos import AztecOO

from dolfin import down_cast, Vector, Matrix
import numpy
from BlockStuff import BlockBase, BlockCompose

class MLPreconditioner(BlockBase):
    def __init__(self, A, pdes=1):
        # create the ML preconditioner
        MLList = {
            #"max levels"                : 30,
#            "ML output"                 : 10,
            "smoother: type"            : "ML symmetric Gauss-Seidel" ,
            #"smoother: sweeps"          : 2,
            #"cycle applications"        : 2,
            #"prec type"                 : "MGW",
            "aggregation: type"         : "Uncoupled" ,
            #"PDE equations"             : pdes,
            "ML validate parameter list": True,
            }
        self.A = A # Prevent matrix being deleted
        self.ml_prec = ML.MultiLevelPreconditioner(down_cast(A).mat(), 0)
        self.ml_prec.SetParameterList(MLList)
        self.ml_agg = self.ml_prec.GetML_Aggregate()
        err = self.ml_prec.ComputePreconditioner()
        if err:
            raise RuntimeError('ComputePreconditioner returned %d'%err)

    def __mul__(self, b):
        if not isinstance(b, Vector):
            return BlockCompose(self, b)
        # apply the ML preconditioner
        x = Vector(len(b))
        err = self.ml_prec.ApplyInverse(down_cast(b).vec(), down_cast(x).vec())
        if err:
            raise RuntimeError('ApplyInverse returned %d'%err)
        return x

    def down_cast(self):
        return self.ml_prec

class AztecSolver(BlockBase):
    def __init__(self, A, tolerance=1e-5, maxiter=300, solver='cg', precond=None):
        self.A = A
        self.solver = getattr(AztecOO, 'AZ_'+solver)
        if isinstance(precond, str):
            self.precond = getattr(AztecOO, 'AZ_'+precond)
        else:
            self.precond = precond
        self.tolerance = tolerance
        self.maxiter = maxiter

    def __mul__(self, b):
        if not isinstance(b, Vector):
            return BlockCompose(self, b)
        x = Vector(len(b))
        solver = AztecOO.AztecOO(down_cast(self.A).mat(), down_cast(x).vec(), down_cast(b).vec())
        #solver.SetAztecDefaults()
        solver.SetAztecOption(AztecOO.AZ_solver, self.solver)
        if self.precond:
            if hasattr(self.precond, 'down_cast'):
                solver.SetPrecOperator(self.precond.down_cast())
            else:
                # doesn't seem to work very well
                solver.SetAztecOption(AztecOO.AZ_precond, self.precond)
                # the following are from the example with precond='dom_decomp'
                solver.SetAztecOption(AztecOO.AZ_subdomain_solve, AztecOO.AZ_ilu)
                solver.SetAztecOption(AztecOO.AZ_overlap, 1)
                solver.SetAztecOption(AztecOO.AZ_graph_fill, 1)

        solver.SetAztecOption(AztecOO.AZ_output, 0)
        solver.Iterate(self.maxiter, self.tolerance)
        return x
