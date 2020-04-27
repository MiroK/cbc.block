from __future__ import absolute_import
from petsc4py import PETSc
import numpy as np


def petsc_vector(vec, bvec):
    '''Return PETSc.Vec representation of block vector.'''
    # NOTE: this is not recasting but different type object with 'same'
    # data
    bvec_array = np.hstack([xi.get_local() for xi in bvec])
    vec.setArray(bvec_array)
    return vec


def block_vector(bvec, vec):
    '''Return block_vec representation of PETSc.Vec.'''
    # NOTE: this is not recasting but different type object with 'same'
    # data
    vec_array = vec.array
    start_index = 0
    for xi in bvec:
        end_index = start_index + xi.size()
        xi.set_local(vec_array[start_index:end_index])
        xi.apply('insert')
        start_index = end_index
    return bvec


def petsc_matrix(A):
    '''
    Represent block_mat as a petsc matrix (by having corrent matrix
    vector product
    '''
    x_size = sum(xi.size() for xi in A.create_vec(1))
    b_size = sum(bj.size() for bj in A.create_vec(0))
    
    A_petsc = PETSc.Mat().createPython([[x_size, ]*2, [b_size, ]*2])
    A_petsc.setPythonContext(WrapAction(A))
    A_petsc.setUp()

    return A_petsc


def petsc_preconditioner(B, mat, A):
    '''
    Represent block_mat as a petsc matrix (by having corrent matrix
    vector product
    '''
    pc = PETSc.PC().create()
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setOperators(mat)   # To get the sizes right
    pc.setPythonContext(WrapAction(B, A))
    pc.setUp()
    
    return pc


class WrapAction(object):
    def __init__(self, A, B=None):
        if B is None:
            self.x_block = A.create_vec(1)
        else:
            self.x_block = B.create_vec(1)
        self.A = A

    def mult(self, mat, x, y):
        '''y = A*x'''
        x_block = block_vector(self.x_block, x)
        y_block = self.A*x_block
        petsc_vector(y, y_block)

    def apply(self, pc, x, y):
        '''y = A*x but with A as the preconditioner'''
        x_block = block_vector(self.x_block, x)
        y_block = self.A*x_block
        petsc_vector(y, y_block)
