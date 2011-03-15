import sys
if 'dolfin' in sys.modules and not 'PyTrilinos' in sys.modules:
    raise RuntimeError('must be imported before dolfin -- add "import PyTrilinos" first in your script')
del sys

from MLPrec import ML
from AztecOO import AztecSolver
from IFPACK import *
from Epetra import *

# To be able to use ML we must instruct Dolfin to use the Epetra backend.
import dolfin
dolfin.parameters["linear_algebra_backend"] = "Epetra"

del dolfin, block_base
