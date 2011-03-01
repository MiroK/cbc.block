from dolfin import *
from dolfin_util import *
import numpy
import time

#set_log_level(WARNING)

# Function spaces, elements

mesh = UnitSquare(16,16)
dim = mesh.topology().dim()


V_u = VectorFunctionSpace(mesh, "CG", 2)
V_p = FunctionSpace(mesh, "CG", 1)
