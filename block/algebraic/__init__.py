import sys
if 'dolfin' in sys.modules and not 'PyTrilinos' in sys.modules:
    raise RuntimeError('must be imported before dolfin -- add "import PyTrilinos" first in your script')
del sys

from MLPrec import ML
from AztecOO import AztecSolver
from IFPACK import *
from Epetra import *

del Vector, block_base
