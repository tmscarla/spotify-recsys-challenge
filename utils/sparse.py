import numpy as np
import scipy.sparse as sp
import operator

def inplace_set_rows_zero_where_sum(X, op, cut):
    scale = np.ones(X.shape[0], dtype=np.float)
    sums = sum_rows(X)
    op = __get_op(op)
    scale = np.where(~op(sums, cut), scale, 0)
    inplace_row_scale(X, scale)
    
def inplace_set_cols_zero_where_sum(X, op, cut):
    scale = np.ones(X.shape[1], dtype=np.float)
    sums = sum_cols(X)
    op = __get_op(op)
    scale = np.where(~op(sums, cut), scale, 0)
    inplace_col_scale(X, scale)

def inplace_set_rows_zero(X, target_rows):
    ids = np.array(target_rows, dtype=np.int)
    scale = np.ones(X.shape[0], dtype=np.float)
    scale[ids] = 0
    inplace_row_scale(X, scale)

def inplace_set_cols_zero(X, target_cols):
    ids = np.array(target_cols, dtype=np.int)
    scale = np.ones(X.shape[1],dtype=np.float)
    scale[ids] = 0
    inplace_col_scale(X, scale)

def inplace_row_scale(X, scale): 
    if isinstance(X, sp.csc_matrix):
        __inplace_csr_col_scale(X.T, scale)
    elif isinstance(X, sp.csr_matrix):
        __inplace_csr_row_scale(X, scale)
    else:
        _raise_typeerror(X)

def inplace_col_scale(X, scale):
    if isinstance(X, sp.csc_matrix):
        __inplace_csr_row_scale(X.T, scale)
    elif isinstance(X, sp.csr_matrix):
        __inplace_csr_col_scale(X, scale)
    else:
        _raise_typeerror(X)
    
def sum_cols(X):
    return np.array(X.sum(axis = 0).A1, dtype=np.float)

def sum_rows(X):
    return np.array(X.sum(axis = 1).A1, dtype=np.float)

def __inplace_csr_col_scale(X, scale):
    assert scale.shape[0] == X.shape[1]
    X.data *= scale.take(X.indices, mode='clip')

def __inplace_csr_row_scale(X, scale):
    assert scale.shape[0] == X.shape[0]
    X.data *= np.repeat(scale, np.diff(X.indptr))

def __get_op(op):
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq,
           '==': operator.eq,
           '!=':operator.ne}
    return ops[op]