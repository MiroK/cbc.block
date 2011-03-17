#!/bin/sh
set -e
export DOLFIN_NOPLOT=1

if which parallel >/dev/null 2>/dev/null; then
    parallel -j +0 --halt-on-error=2 -n 1 python ::: demo-*.py
else
    echo demo-*.py | xargs -n 1 python
fi

# hodge8 is the only one that runs in parallel, since it doesn't use Dirichlet BCs.
mpirun -np 3 python demo-hodge8.py N=4
