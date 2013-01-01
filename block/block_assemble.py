from __future__ import division
from block import *
from block_util import block_tensor, isscalar, wrap_in_list, create_vec_from

def block_assemble(forms, bcs=None, symmetric_mod=None):
    # Depending on the shape of forms, a block_mat or a block_vec is returned.
    forms = block_tensor(forms)
    tensor = block_tensor(forms)
    if bcs is None:
        bcs = [[]]*forms.blocks.shape[0]
    if isinstance(forms, block_vec):
        for i in range(len(forms)):
            tensor[i] = _assemble_vec(forms[i], bcs=bcs[i])
        if symmetric_mod:
            I = block_mat.diag(1, symmetric_mod.blocks.shape[0])
            tensor.allocate(symmetric_mod)
            tensor = (I-symmetric_mod)*tensor
    else:
        assert symmetric_mod is None
        for i in range(forms.blocks.shape[0]):
            for j in range(forms.blocks.shape[1]):
                tensor[i,j] = _assemble_mat(forms[i,j], bcs=bcs[i], diag=(i==j))
    return tensor

def block_symmetric_assemble(forms, bcs):
    # Two block_mats are returned (symmetric and asymmetric parts).
    forms = block_tensor(forms)
    assert len(forms.blocks.shape) == 2
    symm = block_mat(forms.blocks)
    asymm = block_mat(forms.blocks)
    if bcs is None:
        bcs = [[]]*forms.blocks.shape[0]
    for i in range(forms.blocks.shape[0]):
        for j in range(forms.blocks.shape[1]):
            symm[i,j], asymm[i,j] = _symmetric_assemble(forms[i,j], row_bcs=bcs[i], col_bcs=bcs[j])
    return symm, asymm

def _is_form(form):
    from dolfin.cpp import Form as cpp_Form
    from ufl.form import Form as ufl_Form
    return isinstance(form, (cpp_Form, ufl_Form))

def _assemble_mat(form, bcs, diag):
    if _is_form(form):
        from dolfin import assemble, symmetric_assemble
        if diag or not bcs:
            return assemble(form, bcs=bcs)
        else:
            return symmetric_assemble(form, row_bcs=bcs)[0]
    if not bcs:
        return form
    if form != 0:
        raise NotImplementedError("can't set Dirichlet BCs on a non-zero, non-form block")
    if not diag:
        return form
    A = _new_square_matrix(bcs)
    A *= 0
    for bc in wrap_in_list(bcs):
        bc.apply(A)
    return A

def _assemble_vec(form, bcs):
    if _is_form(form):
        from dolfin import assemble
        return assemble(form, bcs=bcs)
    if not bcs:
        return form
    v = create_vec_from(bcs)
    v[:] = form
    for bc in wrap_in_list(bcs):
        bc.apply(v)
    return v

def _symmetric_assemble(form, row_bcs=None, col_bcs=None):
    if isscalar(form):
        ret = _assemble_mat(form, row_bcs, diag=(row_bcs==col_bcs))
        return ret, 0
    else:
        from dolfin import symmetric_assemble
        return symmetric_assemble(form, row_bcs=row_bcs, col_bcs=col_bcs)

def _new_square_matrix(bcs):
    import block.algebraic
    vec = create_vec_from(bcs)
    return block.algebraic.active_backend().create_identity(vec, val=0.0)
