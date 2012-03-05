from __future__ import division
from dolfin import warning

def copy(obj):
    if hasattr(obj, 'copy'):
        return obj.copy()
    else:
        import copy
        try:
            return copy.deepcopy(obj)
        except TypeError:
            warning("Don't know how to make a deep copy of (%d,%d), making shallow copy"%(i,j))
            return copy.copy(obj)
