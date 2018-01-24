from block.algebraic.petsc.type_wrap import *


def petsc_minres(B, A, x, b, tolerance, maxiter, progress, relativeconv=False, shift=0, callback=None):
    '''PETSc's MinRes implementation'''
    # Conversions
    A_petsc = petsc_matrix(A)
    x_petsc, b_petsc = A_petsc.createVecs()
    # Fill
    x_petsc = petsc_vector(x_petsc, x)
    b_petsc = petsc_vector(b_petsc, b)
    if B == 1.0:
        B_petsc = PETSc.PC().create()
        B_petsc.setOperators(A_petsc)
        B_petsc.setType('none')
        B_petsc.setUp()
    else:
        # This is awkward; B is for action, A_petsc is for sizes, A is
        # for block_vec allocation which B cannot do
        B_petsc = petsc_preconditioner(B, A_petsc, A)

    # Solver setup
    ksp = PETSc.KSP().create()
    ksp.setType('minres')
    ksp.setOperators(A_petsc)
    
    if relativeconv:
        ksp.setTolerances(rtol=tolerance, atol=None, divtol=None, max_it=maxiter)
    else:
        ksp.setTolerances(rtol=None, atol=tolerance, divtol=None, max_it=maxiter)

    ksp.setConvergenceHistory()
    ksp.setPC(B_petsc)

    # Solve
    ksp.solve(b_petsc, x_petsc)
    residuals = ksp.getConvergenceHistory()

    # Convert back
    x = block_vector(x, x_petsc)

    return x, residuals, [], []
    
# -------------------------------------------------------------------

if __name__ == '__main__':
    from block import block_assemble
    from dolfin import *

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = [V, Q]
    
    vq = map(TestFunction, W)

    bb = block_assemble(map(lambda v: inner(Constant(1), v)*dx, vq))

    vec = PETSc.Vec().create()
    vec.setSizes((V.dim() + Q.dim(), )*2)
    vec.setUp()

    #print vec.array
    vec = petsc_vector(vec, bb)

    bb0 = bb.copy()
    
    bb = block_vector(bb, vec)

    for x, y in zip(bb, bb0):
        print (x-y).norm('linf'), x.norm('l2')

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    AA = block_assemble([[inner(u, v)*dx, 0], [0, inner(p, q)*dx]])

    x = bb.copy()
    print petsc_minres(B=1, A=AA, x=x, b=bb, tolerance=1E-10, maxiter=100, progress=1, relativeconv=False)
