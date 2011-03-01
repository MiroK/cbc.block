"""This demo illustrates basic usage of block matrices and vectors."""

__author__ = "Kent-Andre Mardal (kent-and@simula.no)"
__date__ = "2008-12-12 -- 2008-12-12"
__copyright__ = "Copyright (C) 2008 Kent-Andre Mardal"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

parameters["linear_algebra_backend"] = "Epetra"

# Create a simple stiffness matrix
mesh = UnitSquare(4, 4)

V = FunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)

v, u = TestFunction(V), TrialFunction(V)
s, t = TestFunction(W), TrialFunction(W)

# Create a block matrix and vector

AA = BlockMatrix(2,2)
AA[0,0] = assemble(v*u*dx)
AA[0,1] = assemble(v*t*dx)
AA[1,0] = assemble(s*u*dx)
AA[1,1] = assemble(s*t*dx)

xx = BlockVector(2)
xx[0] = assemble(100*v*dx)
xx[1] = assemble(100*s*dx)

# Multiply

yy = AA * xx

# Check results

norm = yy.norm("l2")
print "[%d]  ||AAx|| = %f" % (MPI.process_number(), norm)
assert abs(norm - 1.8499437321) < 1e-10
