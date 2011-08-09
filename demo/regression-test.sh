#!/bin/sh
set -e
export DOLFIN_NOPLOT=1

if which parallel >/dev/null 2>/dev/null; then
    parallel -j +0 --halt-on-error=2 -n 1 python ::: *.py
else
    echo *.py | xargs -n 1 python
fi

# hodge8 is the only one that runs in parallel, since it doesn't use Dirichlet BCs.
mpirun -np 3 python fenics-book/hodge.py N=4
mpirun -np 3 python parallelmixedpoisson.py
