### Material parameters

lmbda = Constant(1e5)
mu    = Constant(1e5)

S = Constant(1e-2)
alpha = Constant(1.0)
Lambda = Constant( numpy.diag([.02]*(dim-1)+[.001]) )

t_n = Constant([0.0]*dim)
f_n = Constant([0.0]*dim)

dt = Constant(.02)
T = 0.02

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

bm_source = Expression('2e5*sin(2*pi*t)')
c = 0.25
h = mesh.hmin()
fluid_source_domain = compile_subdomains('between({min},x[0],{max}) && between({min},x[1],{max})'
                                         .format(min=c-h, max=c+h))

def bc_barry_mercer():

    bc_u_tangent_ew  = DirichletBC(V_u.sub(1), 0,         boundary.ew)
    bc_u_tangent_tb  = DirichletBC(V_u.sub(0), 0,         boundary.tb)
    bc_p_source      = DirichletBC(V_p,        bm_source, fluid_source_domain)
    bc_p_drained_all = DirichletBC(V_p,        0,         boundary.all)

    return ([bc_u_tangent_ew, bc_u_tangent_tb], [bc_p_source, bc_p_drained_all])

topload_source = Expression("-sin(2*t*pi)*sin(x[0]*pi/2)/3")
bc_u_symm_ew = DirichletBC(V_u.sub(0),     0,       boundary.ew)
bc_u_symm_ns = DirichletBC(V_u.sub(1),     0,       boundary.ns)
bc_u_bedrock = DirichletBC(V_u,            [0]*dim, boundary.bottom)
bc_u_topload = DirichletBC(V_u.sub(dim-1), Expression("-0.1*sin(x[0])"), boundary.top)
bc_u_topload = DirichletBC(V_u.sub(dim-1), topload_source, boundary.top)
bc_p_drained = DirichletBC(V_p,            0,       boundary.top)
bc_p_drained_ew = DirichletBC(V_p,            0,       boundary.west)

bc_p_drained_source = DirichletBC(V_p, 0, fluid_source_domain)

#bcs = bc_barry_mercer()
#bcs = [[bc_u_symm_ew, bc_u_symm_ns, bc_u_bedrock, bc_u_topload], [bc_p_drained]]
bcs = [[bc_u_topload, bc_u_bedrock], [bc_p_drained_source]]

update.set_args(displacement={'mode': 'displacement', 'wireframe': True},
                volumetric={'functionspace': FunctionSpace(mesh, "DG", 0)},
                velocity={'functionspace': VectorFunctionSpace(mesh, "DG", 0)}
                )


def check_symmetric(M, N, name=None):
    M = M.array()
    N = N.array()
    if numpy.all(abs(M-N.T)<1e-8):
        print '%s: Symmetric'%name
    else:
        print '%s is NOT symmetric'%name

def check_definite(M, name):
    eigs = numpy.linalg.eig(M.array())[0]
    print '%s: eigs in (%g, %g)' % (name, min(eigs), max(eigs))

