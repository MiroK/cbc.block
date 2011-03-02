from __future__ import division

__author__  = "Joachim B Haga <jobh@simula.no>"
__date__    = "2011-01"
__license__ = "Any"

from MLPrec import *
#========
# Define mesh, spaces
execfile('biot_head.py')
#========
from BlockStuff import *
from BlockSolvers import *

parameters["linear_algebra_backend"] = "Epetra"

v, omega = TrialFunction(V_u), TestFunction(V_u)
q, phi   = TrialFunction(V_p), TestFunction(V_p)

u_prev = Function(V_u)
p_prev = Function(V_p)

#========
# Define forms, material parameters, boundary conditions, etc.
execfile('biot_common.py')
#========

# Assemble the matrices
A   = assemble(a00)
B   = assemble(a01)
C   = assemble(a10)
D   = assemble(a11)
b_p = assemble(L1)

# Insert the matrices into blocks

AA = BlockOperator([[A, B],
                    [C, D]])
bb = BlockVector([0, b_p])

# Apply boundary conditions

bcs = BlockBC(bcs)
bcs.apply(AA, bb, save_A=True)

Ap = ML(A)
Ainv = ConjGrad(A, precond=Ap, name='Ainv')

Sa = AA.schur_approximation(scale=-1)
Sp = ML(Sa)

S = C*Ainv*B-D
Sinv = ConjGrad(S, precond=Sp, name='Sinv')

#=====================
#AAinv = BlockOperator([[Ainv, B],
#                       [C, -Sinv]]).scheme('sgs')
AApre = blockop([[Ap, B],
                 [C, -Sp]]).scheme('jac')
AAinv = TFQMR(AA, precond=AApre)
#=====================

u = Function(V_u)
p = Function(V_p)

t = 0.0
while t <= T:
    print "Time step %f" % t

    topload_source.t = t

    bb[0].zero()
    bb[1] = assemble(L1)
    bcs.apply(bb)

    before = time.time()
    xx = AAinv * bb
    after = time.time()

    u.vector()[:] = xx[0]
    p.vector()[:] = xx[1]

    print 'Residual of solution: %.1e [%.2f seconds]'%((AA*xx-bb).norm(), after-before)


    update(time=t,
           displacement=u,
           pressure=p,
           velocity=v_D(p),
#           volumetric=tr(sigma(u)),
           )

    u_prev.vector()[:] = u.vector()
    p_prev.vector()[:] = p.vector()
    t += float(dt)

interactive()
print "Finished normally"
