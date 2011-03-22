import sys

class object_pool(object):
    """Manage a free-list of objects. The objects are automatically made
    available as soon as they are deleted by the caller. The assumption is that
    any operation is repeated a number of times (think iterative solvers), so
    that if N objects are needed simultaneously then soon N objects are needed
    again. Thus, objects managed by this pool are not deleted until the owning
    object (typically a Matrix) is deleted.
    """
    def __init__(self):
        self.all = set()
        self.free = []

    def add(self, obj):
        self.all.add(obj)

    def get(self):
        self.collect()
        return self.free.pop()

    def collect(self):
        for obj in self.all:
            if sys.getrefcount(obj) == 3:
                # 3 references: self.all, obj, getrefcount() parameter
                self.free.append(obj)

def vec_pool(func):
    """Decorator for create_vec, which creates a per-object pool of (memoized)
    returned vectors.
    """
    from collections import defaultdict
    def pooled_create_vec(self, dim=1):
        if not hasattr(self, '_vec_pool'):
            self._vec_pool = defaultdict(object_pool)
        try:
            vec = self._vec_pool[dim].get()
        except IndexError:
            vec = func(self, dim)
            self._vec_pool[dim].add(vec)
        return vec
    pooled_create_vec.__doc__ = func.__doc__
    return pooled_create_vec
