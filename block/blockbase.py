class BlockBase(object):
    # Convenience methods
    def __mul__(self, other):
        if not isinstance(other, (BlockVector, Vector)):
            return BlockCompose(self, other)
        return self.matvec(other)

    def __rmul__(self, other):
        return BlockCompose(other, self)

    def __neg__(self):
        return BlockCompose(-1, self)

class BlockThingy(BlockBase):
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
