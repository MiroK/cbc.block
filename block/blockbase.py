from __future__ import division

import numpy

class blockbase(object):
    # Convenience methods
    def __mul__(self, other):
        from blockcompose import blockcompose
        from blockvector import blockvec
        from dolfin import Vector
        if not isinstance(other, (blockvec, Vector)):
            return blockcompose(self, other)
        return self.matvec(other)

    def __rmul__(self, other):
        from blockcompose import blockcompose
        return blockcompose(other, self)

    def __neg__(self):
        from blockcompose import blockcompose
        return blockcompose(-1, self)

class blockcontainer(blockbase):
    def __init__(self, mn=None, blocks=None):
        if mn:
            self.blocks = numpy.ndarray(mn, dtype=numpy.object)
            if blocks is not None:
                self.blocks[:] = blocks
        elif blocks:
            self.blocks = numpy.array(blocks, dtype=numpy.object)
        else:
            raise TypeError('must pass at least one argument')

    def __setitem__(self, key, val):
        self.blocks[key] = val
    def __getitem__(self, key):
        try:
            return self.blocks[key]
        except IndexError, e:
            raise IndexError(str(e) + ' at ' + str(key) + ' -- incompatible block structure')
    def __len__(self):
        return len(self.blocks)
    def __iter__(self):
        return self.blocks.__iter__()
    def __str__(self):
        return str(type(self))+': '+str(self.blocks)
