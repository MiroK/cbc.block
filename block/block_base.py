from __future__ import division

import numpy

class block_base(object):
    # Convenience methods
    def __mul__(self, other):
        from block_compose import block_compose
        from block_vec import block_vec
        from dolfin import Vector
        if not isinstance(other, (block_vec, Vector)):
            return block_compose(self, other)
        return self.matvec(other)

    def __rmul__(self, other):
        from block_compose import block_compose
        return block_compose(other, self)

    def __neg__(self):
        from block_compose import block_compose
        return block_compose(-1, self)

    def __add__(self, other):
        from block_compose import block_add
        return block_add(self, other)

    def __radd__(self, other):
        return other.__add__(self)

    def __sub__(self, other):
        from block_compose import block_sub
        return block_sub(self, other)

    def __rsub__(self, other):
        from block_compose import block_sub
        return block_sub(other, self)

class block_container(block_base):
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
        return '<%s %s:\n%s>'%(self.__class__.__name__,
                               'x'.join(map(str, self.blocks.shape)),
                               str(self.blocks))
