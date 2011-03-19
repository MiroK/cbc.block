from __future__ import division

"""Utility functions for plotting, boundaries, etc."""

import os
import dolfin as d
import time as timer

class BoxBoundary(object):
    def __init__(self, mesh):
        c = mesh.coordinates()
        self.c_min, self.c_max = c.min(0), c.max(0)
        dim = len(self.c_min)

        sd = self._compile(west  = self._boundary(0, self.c_min) if dim>1 else '0',
                           east  = self._boundary(0, self.c_max) if dim>1 else '0',
                           south = self._boundary(1, self.c_min) if dim>2 else '0',
                           north = self._boundary(1, self.c_max) if dim>2 else '0',
                           bottom= self._boundary(dim-1, self.c_min),
                           top   = self._boundary(dim-1, self.c_max),
                           ew    = self._boundary(0) if dim>1 else '0',
                           ns    = self._boundary(1) if dim>1 else '0',
                           tb    = self._boundary(dim-1),
                           all   = 'on_boundary')
        for name,subdomain in sd:
            setattr(self, name, subdomain)

    def _boundary(self, idx, coords=None):
        if coords is not None:
            return 'on_boundary && near(x[{idx}], {coord})' \
                .format(idx=idx, coord=coords[idx])
        else:
            return 'on_boundary && (near(x[{idx}], {min}) || near(x[{idx}], {max}))' \
                .format(idx=idx, min=self.c_min[idx], max=self.c_max[idx])

    def _compile(self, **kwargs):
        # The same ordering of keys() and values() is guaranteed.
        names = kwargs.keys()
        compiled = map(d.compile_subdomains, kwargs.values())
        return [(names[i],compiled[i]) for i in range(len(names))]

class update():
    """Plot and save given functional(s). Example:
    u = problem.solve()
    update.set_args(displacement={'mode': 'displacement'})
    update(displacement=u, volumetric=tr(sigma(u)))
    """
    files = {}
    plots = {}
    kwargs = {}
    projectors = {}

    def project(self, f, V):
        if V is None:
            V = d.fem.project.func_globals['_extract_function_space'](f)
        key = str(V)
        v = d.TestFunction(V)
        if not key in self.projectors:
            # Create mass matrix
            u = d.TrialFunction(V)
            a = d.inner(v,u) * d.dx
            self.projectors[key] = (d.assemble(a), d.Function(V))

        A,Pf = self.projectors[key]
        b = d.assemble(d.inner(v,f) * d.dx)

        # Solve linear system for projection
        d.solve(A, Pf.vector(), b, "cg")

        return Pf


    def set_args(self, **kwargs):
        """Set additional kwargs to pass to plot for a given name.

        In addition to the kwargs for plot, these are accepted:
        'plot' (bool)                   -- plot to screen [True]
        'save' (bool)                   -- save to file [True]
        'functionspace' (FunctionSpace) -- space to project to [CG(1)]"""
        self.kwargs.update(kwargs)

    def save_to_file(self, name, data, time):
        if not os.path.exists('data'):
            os.mkdir('data')
        if not name in self.files:
            self.files[name] = d.File('data/%s.pvd'%name)
        if time is not None:
            self.files[name] << (data, time)
        else:
            self.files[name] << data

    def plot(self, name, data, time):
        if not name in self.plots:
            kwargs = self.kwargs.get(name, {})
            self.plots[name] = d.plot(data, title=name, size=(400,400),
                                      axes=True, warpscalar=False,
                                      **kwargs)
        else:
            self.plots[name].update(data)

    def __call__(self, time=None, **functionals):
        for name,func in sorted(functionals.iteritems()):
            args = self.kwargs.get(name, {})
            if 'functionspace' in args or not isinstance(func, d.Function):
                func = self.project(func, args.get('functionspace'))
            if hasattr(func, 'rename'):
                func.rename(name, name)
            if args.get('plot', True):
                self.plot(name, func, time)
            if args.get('save', True):
                self.save_to_file(name, func, time)

update = update() # singleton
