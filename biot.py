from __future__ import division

__author__  = "Joachim B Haga <jobh@simula.no>"
__date__    = "2011"
__license__  = "GNU LGPL Version 2.1"

# Since we use ML from Trilinos, we must import PyTrilinos before any dolfin
# modules. This works around a bug with MPI initialisation/destruction order.
import PyTrilinos

from block import *
from block.iterative import *
from block.algebraic import *
from dolfin import *
from dolfin_util import *
import numpy

# To be able to use ML we must instruct Dolfin to use the Epetra backend.
parameters["linear_algebra_backend"] = "Epetra"

# Function spaces, elements

mesh = UnitSquare(16,16)
dim = mesh.topology().dim()

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)

v, omega = TrialFunction(V), TestFunction(V)
q, phi   = TrialFunction(W), TestFunction(W)

u_prev = Function(V)
p_prev = Function(W)

#========
# Define forms, material parameters, boundary conditions, etc.

### Material parameters

lmbda = Constant(1e5)
mu    = Constant(1e5)

S = Constant(1e-2)
alpha = Constant(1.0)
Lambda = Constant( numpy.diag([.02]*(dim-1)+[.001]) )

t_n = Constant([0.0]*dim)
f_n = Constant([0.0]*dim)

dt = Constant(.02)
T = 0.2

Q = Constant(0)

def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(grad(v))*Identity(dim)
def v_D(q):
    return -Lambda*grad(q)
def b(w,r):
    return - alpha * r * div(w)

a00 = inner(grad(omega), sigma(v)) * dx
a01 = b(omega,q) * dx
a10 = b(v,phi) * dx
a11 = -(S*phi*q - dt*inner(grad(phi),v_D(q))) * dx

L1 = b(u_prev,phi) * dx - (Q*dt + S*p_prev)*phi * dx

boundary = BoxBoundary(mesh)

c = 0.25
h = mesh.hmin()
fluid_source_domain = compile_subdomains('{min}<x[0] && x[0]<{max} && {min}<x[1] && x[1]<{max}'
                                         .format(min=c-h, max=c+h))
topload_source      = Expression("-sin(2*t*pi)*sin(x[0]*pi/2)/3")

bc_u_bedrock        = DirichletBC(V,            [0]*dim,        boundary.bottom)
bc_u_topload        = DirichletBC(V.sub(dim-1), topload_source, boundary.top)
bc_p_drained_source = DirichletBC(W,            0,              fluid_source_domain)

bcs = [[bc_u_topload, bc_u_bedrock], [bc_p_drained_source]]

update.set_args(displacement={'mode': 'displacement', 'wireframe': True},
                volumetric={'functionspace': FunctionSpace(mesh, "DG", 0)},
                velocity={'functionspace': VectorFunctionSpace(mesh, "DG", 0)}
                )

# Assemble the matrices
A   = assemble(a00)
B   = assemble(a01)
C   = assemble(a10)
D   = assemble(a11)
b_p = assemble(L1)

# Insert the matrices into blocks

AA = block_mat([[A, B],
                [C, D]])
bb = block_vec([0, b_p])

# Apply boundary conditions

bcs = block_bc(bcs)

# Must set save_A in order to do symmetric modification of bb in time loop
bcs.apply(AA, bb, save_A=True)

# Create preconditioner -- a generalised block Jacobi preconditioner, where the
# (2,2) approximates the pressure Schur complement.

Ap = ML(A)

S = C*InvDiag(A)*B-D
Sp = ML(explicit(S))

AApre = block_mat([[Ap, 0],
                   [0, -Sp]])


AAinv = SymmLQ(AA, precond=AApre, show=2, tolerance=1e-10)
#=====================

u = Function(V)
p = Function(W)

t = 0.0
while t <= T:
    print "Time step %f" % t

    topload_source.t = t

    bb[0].zero()
    bb[1] = assemble(L1)
    bcs.apply(bb)

    U, P = AAinv * bb

    u.vector()[:] = U
    p.vector()[:] = P

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
