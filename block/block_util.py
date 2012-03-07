from __future__ import division

def copy(obj):
    """Return a deep copy of the object"""
    if hasattr(obj, 'copy'):
        return obj.copy()
    else:
        import copy
        try:
            return copy.deepcopy(obj)
        except TypeError:
            from dolfin import warning
            warning("Don't know how to make a deep copy of (%d,%d), making shallow copy"%(i,j))
            return copy.copy(obj)

def block_tensor(obj):
    """Return either a block_vec or a block_mat, depending on the shape of the object"""
    from block import block_mat, block_vec
    import numpy
    if isinstance(obj, (block_mat, block_vec)):
        return obj
    blocks = numpy.array(obj)
    if len(blocks.shape) == 2:
        return block_mat(blocks)
    elif len(blocks.shape) == 1:
        return block_vec(blocks)
    else:
        raise RuntimeError("Not able to create block container of rank %d"%len(blocks.shape))
