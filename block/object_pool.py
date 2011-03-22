class object_pool(object):
    """Manage a free-list of objects"""

    def __init__(self, capacity=2):
        self.capacity = capacity
        self.all = []
        self.free = []

    def push(self, obj):
        if hash(obj) in map(hash, self.all):
            raise ValueError, 'object added twice'
        self.all.append(obj)

    def pop(self):
        self.collect()
        return self.free.pop()

    def collect(self):
        import sys
        # walk through list backwards so that deletions don't mess up numbering
        for i in range(len(self.all)-1, -1, -1):
            # the list holds one reference, the temporary holds one
            if sys.getrefcount(self.all[i]) == 2:
                if len(self.free) < self.capacity:
                    self.free.append(self.all[i])
                else:
                    del self.all[i]

def vec_pool(func):
    """decorator for create_vec"""
    from collections import defaultdict
    def pooled_create_vec(self, dim=1):
        if not hasattr(self, '_vec_pool'):
            from block.object_pool import object_pool
            self._vec_pool = defaultdict(object_pool)
        try:
            vec = self._vec_pool[dim].pop()
        except IndexError:
            vec = func(self,dim)
            self._vec_pool[dim].push(vec)
        return vec
    pooled_create_vec.__doc__ = func.__doc__
    return pooled_create_vec
