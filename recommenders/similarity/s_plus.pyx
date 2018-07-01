"""
    author: Simone Boglio
    mail: simone.boglio@mail.polimi.it
"""

import cython
import numpy as np
import scipy.sparse as sp
import tqdm

from sklearn.utils.sparsefuncs import inplace_column_scale
from sklearn.utils.sparsefuncs import inplace_row_scale

from scipy.sparse.sputils import get_index_dtype

from cython.operator import dereference
from cython.parallel import parallel, prange
from cython import address
from cython import float

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp cimport bool

cdef extern from "s_plus.h" namespace "similarity" nogil:
    cdef cppclass TopK[Index, Value]:
        TopK(size_t K)
        vector[pair[Value, Index]] results

    cdef cppclass SparseMatrixMultiplier[Index, Value]:
        SparseMatrixMultiplier(Index user_count,
                               Value * detXlessY, Value * detYlessX,
                               Value * detX, Value * detY,
                               Value n,
                               Value l1, Value l2,
                               Value t1, Value t2,
                               Value c1, Value c2,
                               Value shrink, Value threshold)
        void add(Index index, Value value)
        void setRow(Index index)
        void foreach[Function](Function & f)

cdef extern from "coo.h" nogil:
    void coo_tocsr64(int n_row,int n_col,long nnz,int Ai[],int Aj[],float Ax[],long Bp[],long Bj[],float Bx[])
    void coo_tocsr32(int n_row,int n_col,int nnz,int Ai[],int Aj[],float Ax[],int Bp[],int Bj[],float Bx[])


@cython.boundscheck(False)
@cython.wraparound(False)
def s_plus(
    items, users,
    weight_pop_items='none' , weight_pop_users='none',
    weight_feature_items='none', weight_feature_users='none',
    float p1=0, float p2=0,
    float w1=0, float w2=0,
    bool normalization = True,
    float l1=0.5, float l2=0.5,
    float t1=1, float t2=1,
    float c1=0.5,float c2=0.5,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False,
    int num_threads=0):  

    assert(items.shape[1]==users.shape[0])
    assert(len(weight_pop_items)==items.shape[0] or weight_pop_items in ('none','sum','ln','log'))
    assert(len(weight_pop_users)==users.shape[1] or weight_pop_users in ('none','sum','ln','log'))
    assert(len(weight_feature_items)==items.shape[1] or weight_feature_items in ('none','sum','ln','log'))
    assert(len(weight_feature_users)==users.shape[0] or weight_feature_users in ('none','sum','ln','log'))
    assert(target_items is None or len(target_items)<=items.shape[0])
    assert(verbose==True or verbose==False)
    assert(save_memory==True or save_memory==False)

    # build target items (only the row that must be computed)
    if target_items is None:
        target_items=np.arange(items.shape[0],dtype=np.int32)
    cdef int[:] targets = np.array(target_items,dtype=np.int32)
    cdef int n_targets = targets.shape[0]

    # progress bar
    progress = tqdm.tqdm(total=n_targets, disable=not verbose)
    progress.desc = 'Preprocessing'
    progress.refresh()

    # be sure to use csr matrixes
    items = items.tocsr()
    users = users.tocsr()

    # eliminates zeros to avoid 0 division and get right values of popularities and feature weights
    # note: is an implace operation implemented for csr matrix in the sparse package
    items.eliminate_zeros()
    users.eliminate_zeros()

    # usefull variables
    cdef int item_count = items.shape[0]
    cdef int user_count = users.shape[1]
    cdef int i, u, t, index1, index2
    cdef long index3
    cdef float v1
    cdef float n

    #set normalization
    if normalization==True: n=1
    else: n=0

    #save original data
    old_items, old_users = items.data, users.data

    # if binary use set theorhy otherwise copy data and use float32
    if binary:
        items.data, users.data = np.ones(items.data.shape[0], dtype= np.float32), np.ones(users.data.shape[0], dtype=np.float32)
    else:
        items.data, users.data = np.array(items.data, dtype=np.float32), np.array(users.data, dtype=np.float32)

    #START PREPROCESSING

    # build popularities 
    if isinstance(weight_pop_items,(list,np.ndarray)) and p1!=0: 
        weight_pop_items = np.power(weight_pop_items, -p1, dtype=np.float32)    
    elif weight_pop_items!='none' and p1!=0:
        if weight_pop_items == 'sum':
            weight_pop_items = np.array(items.sum(axis = 1).A1, dtype=np.float32)
        elif weight_pop_items == 'log':
            weights = items.sum(axis = 1).A1
            weight_pop_items = np.log10(weights, where=(weights!=0), dtype=np.float32)
        elif weight_pop_items == 'ln':
            weights = items.sum(axis = 1).A1
            weight_pop_items = np.log(weights, where=(weights!=0), dtype=np.float32)
        weight_pop_items = np.power(weight_pop_items, -p1, dtype=np.float32)
    
    if isinstance(weight_pop_users,(list,np.ndarray)) and p2!=0:
        weight_pop_users = np.power(weight_pop_users, -p2, dtype=np.float32)
    elif weight_pop_users!='none' and p2!=0:
        if weight_pop_users == 'sum':
            weight_pop_users = np.array(users.sum(axis = 0).A1, dtype=np.float32)
        elif weight_pop_users == 'log':
            weights = users.sum(axis = 0).A1
            weight_pop_users = np.log10(weights, where=(weights!=0), dtype=np.float32)
        elif weight_pop_users == 'ln':
            weights = users.sum(axis = 0).A1
            weight_pop_users = np.log(weights, where=(weights!=0), dtype=np.float32)
        weight_pop_users = np.power(weight_pop_users, -p2, dtype=np.float32)

    # build feature weights
    if isinstance(weight_feature_items,(list,np.ndarray)) and w1!=0:
        weight_feature_items = np.power(weight_feature_items, w1, dtype=np.float32)
    elif weight_feature_items!='none' and w1!=0:
        if weight_feature_items == 'sum':
            weight_feature_items = np.array(items.sum(axis = 0).A1, dtype=np.float32)
        elif weight_feature_items == 'log':
            weights = items.sum(axis = 0).A1
            weight_feature_items = np.log10(weights, where=(weights!=0), dtype=np.float32)
        elif weight_feature_items == 'ln':
            weights = items.sum(axis = 0).A1
            weight_feature_items = np.log(weights, where=(weights!=0), dtype=np.float32)
        weight_feature_items = np.power(weight_feature_items, w1, dtype=np.float32)

    if isinstance(weight_feature_users,(list,np.ndarray)) and w2!=0:
        weight_feature_users = np.power(weight_feature_users, w2, dtype=np.float32)
    elif weight_feature_users!='none' and w2!=0:
        if weight_feature_users == 'sum':
            weight_feature_users = np.array(users.sum(axis = 1).A1, dtype=np.float32)
        elif weight_feature_users == 'log':
            weights = users.sum(axis = 1).A1
            weight_feature_users = np.log10(weights, where=(weights!=0), dtype=np.float32)
        elif weight_feature_users == 'ln':
            weights = users.sum(axis = 1).A1
            weight_feature_users = np.log(weights, where=(weights!=0), dtype=np.float32)
        weight_feature_users = np.power(weight_feature_users, w2, dtype=np.float32)

    # now apply popularities and feature weights
    if isinstance(weight_pop_items, np.ndarray) and p1!=0:
        inplace_row_scale(items, weight_pop_items)
    if isinstance(weight_pop_users, np.ndarray) and p2!=0:
        inplace_column_scale(users, weight_pop_users) 
    if isinstance(weight_feature_items, np.ndarray) and w1!=0:
        inplace_column_scale(items, weight_feature_items)
    if isinstance(weight_feature_users, np.ndarray) and w2!=0:
        inplace_row_scale(users, weight_feature_users)

    del weight_feature_items, weight_feature_users
    del weight_pop_items, weight_pop_users

    #END OF PREPROCESSING

    # be sure to use csr matrixes
    items = items.tocsr()
    users = users.tocsr()

    # build the data terms 
    cdef float[:] item_data = np.array(items.data, dtype=np.float32)
    cdef float[:] user_data = np.array(users.data, dtype=np.float32)

    # build indices and indptrs
    cdef int[:] item_indptr = np.array(items.indptr, dtype=np.int32), item_indices = np.array(items.indices, dtype=np.int32)
    cdef int[:] user_indptr = np.array(users.indptr, dtype=np.int32), user_indices = np.array(users.indices, dtype=np.int32)

    # build normalization terms  for tversky and cosine
    cdef float[:] det_x_tversky
    cdef float[:] det_y_tversky
    cdef float[:] det_x_cosine
    cdef float[:] det_y_cosine 

    if normalization and l1!=0:
        det_x_tversky = np.array(items.power(2).sum(axis = 1).A1, dtype=np.float32)
        det_y_tversky = np.array(users.power(2).sum(axis = 0).A1, dtype=np.float32)
    else:
        det_x_tversky = np.array([],dtype=np.float32)
        det_y_tversky = np.array([],dtype=np.float32)

    if normalization and l2!=0:
        det_x_cosine = np.power(items.power(2).sum(axis = 1).A1, c1, dtype=np.float32)
        det_y_cosine = np.power(users.power(2).sum(axis = 0).A1, c2, dtype=np.float32)
    else:
        det_x_cosine = np.array([],dtype=np.float32)
        det_y_cosine = np.array([],dtype=np.float32)

    #restore original data terms
    items.data, users.data = old_items, old_users

    #set progress bar
    cdef int counter = 0
    cdef int * counter_add = address(counter)
    cdef int verb
    if n_targets<=5000 or verbose==False: verb = 0
    else: verb = 1

    
    # structures for multiplications
    cdef SparseMatrixMultiplier[int, float] * neighbours
    cdef TopK[int, float] * topk
    cdef pair[float, int] result

    # triples of output
    cdef float[:] values = np.zeros(n_targets * k, dtype=np.float32)
    cdef int[:] rows = np.zeros(n_targets * k, dtype=np.int32)
    cdef int[:] cols = np.zeros(n_targets * k, dtype=np.int32)

    progress.desc = 'Allocate memory per threads'
    progress.refresh()
    with nogil, parallel(num_threads=num_threads):
        # allocate memory per thread
        neighbours = new SparseMatrixMultiplier[int, float](user_count,
                                                            &det_x_tversky[0], &det_y_tversky[0],
                                                            &det_x_cosine[0], &det_y_cosine[0],
                                                            n, 
                                                            l1, l2,
                                                            t1, t2,
                                                            c1, c2,
                                                            shrink, threshold)
        topk = new TopK[int, float](k)
        try:
            for i in prange(n_targets, schedule='dynamic', chunksize=100):
                #progress bar
                if verb==1:
                    counter_add[0]=counter_add[0]+1 
                    if counter_add[0]%(n_targets/500)==0:
                        with gil:
                            progress.desc = 'Computing'
                            progress.n = counter_add[0]
                            progress.refresh()
                #compute row
                t = targets[i]
                neighbours.setRow(i)
                for index1 in range(item_indptr[t], item_indptr[t+1]):
                    u = item_indices[index1]
                    v1 = item_data[index1]
                    for index2 in range(user_indptr[u], user_indptr[u+1]):
                        neighbours.add(user_indices[index2], user_data[index2] * v1)
                topk.results.clear()
                neighbours.foreach(dereference(topk))
                index3 = k * i
                for result in topk.results:
                    rows[index3] = t
                    cols[index3] = result.second
                    values[index3] = result.first
                    index3 = index3 + 1

        finally:
            del neighbours
            del topk

    progress.n = n_targets
    progress.refresh()
    del det_x_cosine, det_y_cosine, det_x_tversky, det_y_tversky
    del item_data, item_indices, item_indptr
    del targets

    cdef int M,N
    cdef float [:] data
    cdef int [:] indices32, indptr32
    cdef long [:] indices64, indptr64

    if save_memory:
        progress.desc = 'Build coo matrix'
        progress.refresh()
        res = sp.coo_matrix((values, (rows, cols)),shape=(item_count, user_count), dtype=np.float32)
        del values, rows, cols
    else:
        progress.desc = 'Build csr matrix'
        progress.refresh()
        M = item_count
        N = user_count
        idx_dtype = get_index_dtype(maxval= max(n_targets*k,long(N)))
        if idx_dtype==np.int32:
            indptr32 = np.empty(M + 1, dtype=np.int32)
            indices32 = np.empty(n_targets * k, dtype=np.int32)
            data = np.empty(n_targets * k, dtype=np.float32)
            coo_tocsr32(M, N, n_targets*k, &rows[0], &cols[0], &values[0], &indptr32[0], &indices32[0], &data[0])
            del values, rows, cols
            res = sp.csr_matrix((data, indices32, indptr32) ,shape=(item_count, user_count), dtype=np.float32)
            del indptr32,indices32
        elif idx_dtype==np.int64:
            indptr64 = np.empty(M + 1, dtype=np.int64)
            indices64 = np.empty(n_targets * k, dtype=np.int64)
            data = np.empty(n_targets * k, dtype=np.float32)
            coo_tocsr64(M, N, n_targets*k, &rows[0], &cols[0], &values[0], &indptr64[0], &indices64[0], &data[0])
            del values, rows, cols
            res = sp.csr_matrix((data, indices64, indptr64) ,shape=(item_count, user_count), dtype=np.float32)
            del indptr64,indices64
        del data
        progress.desc = 'Remove zeros'
        progress.refresh()
        res.eliminate_zeros()

    
    progress.desc = 'Done'
    progress.refresh()    
    progress.close()
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def dot_product(
    m1, m2,
    unsigned int k=100, float threshold=0,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    return s_plus(
        m1, m2,
        weight_pop_items='none' , weight_pop_users='none',
        weight_feature_items='none', weight_feature_users='none',
        p1=0, p2=0,
        w1=0, w2=0,
        normalization=False,
        l1=0, l2=0,
        t1=0, t2=0,
        c1=0, c2=0,
        k=k, shrink=0, threshold=threshold,
        binary=False,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory) 


@cython.boundscheck(False)
@cython.wraparound(False)
def dot_product_similarity(
    m1, m2,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    return s_plus(
        m1, m2,
        weight_pop_items='none' , weight_pop_users='none',
        weight_feature_items='none', weight_feature_users='none',
        p1=0, p2=0,
        w1=0, w2=0,
        normalization=False,
        l1=0, l2=0,
        t1=0, t2=0,
        c1=0, c2=0,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory) 

@cython.boundscheck(False)
@cython.wraparound(False)
def cosine_similarity(
    m1, m2,
    float alpha=0.5,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    return s_plus(
        m1, m2,
        weight_pop_items='none' , weight_pop_users='none',
        weight_feature_items='none', weight_feature_users='none',
        p1=0, p2=0,
        w1=0, w2=0,
        normalization=True,
        l1=0, l2=1,
        t1=0, t2=0,
        c1=alpha, c2=1-alpha,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory)

@cython.boundscheck(False)
@cython.wraparound(False)
def tversky_similarity(
    m1, m2,
    float alpha=1,float beta=1,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    return s_plus(
        m1, m2,
        weight_pop_items='none' , weight_pop_users='none',
        weight_feature_items='none', weight_feature_users='none',
        p1=0, p2=0,
        w1=0, w2=0,
        normalization=True,
        l1=1, l2=0,
        t1=alpha, t2=beta,
        c1=0, c2=0,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory) 

@cython.boundscheck(False)
@cython.wraparound(False)
def jaccard_similarity(
    m1, m2,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    return s_plus(
        m1, m2,
        weight_pop_items='none' , weight_pop_users='none',
        weight_feature_items='none', weight_feature_users='none',
        p1=0, p2=0,
        w1=0, w2=0,
        normalization=True,
        l1=1, l2=0,
        t1=1, t2=1,
        c1=0, c2=0,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory) 

@cython.boundscheck(False)
@cython.wraparound(False)
def dice_similarity(
    m1, m2,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    return s_plus(
        m1, m2,
        weight_pop_items='none' , weight_pop_users='none',
        weight_feature_items='none', weight_feature_users='none',
        p1=0, p2=0,
        w1=0, w2=0,
        normalization=True,
        l1=1, l2=0,
        t1=0.5, t2=0.5,
        c1=0, c2=0,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory)


@cython.boundscheck(False)
@cython.wraparound(False)
def p3alpha_similarity(
    p_iu, p_ui,
    weight_pop_m1='none' , weight_pop_m2='none',
    float alpha = 1,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    m = s_plus(
        p_iu, p_ui,
        weight_pop_items=weight_pop_m1 , weight_pop_users=weight_pop_m2,
        weight_feature_items='none', weight_feature_users='none',
        p1=0, p2=0,
        w1=0, w2=0,
        normalization=False,
        l1=0, l2=0,
        t1=0, t2=0,
        c1=0, c2=0,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory)
    m.data = np.power(m.data, alpha)
    return m

@cython.boundscheck(False)
@cython.wraparound(False)
def rp3beta_eurm(
    p_ui, p3alpha_similarity,
    weight_pop='none',
    float beta=1,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    return s_plus(
        p_ui, p3alpha_similarity,
        weight_pop_items='none' , weight_pop_users=weight_pop,
        weight_feature_items='none', weight_feature_users='none',
        p1=0, p2=beta,
        w1=0, w2=0,
        normalization=False,
        l1=0, l2=0,
        t1=0, t2=0,
        c1=0, c2=0,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory)

@cython.boundscheck(False)
@cython.wraparound(False)
def feature_weight_similarity(
    m1, m2,
    weight_feature_m1='sum' , weight_feature_m2='sum',
    float w1=1, float w2=1,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    return s_plus(
        m1, m2,
        weight_pop_items='none' , weight_pop_users='none',
        weight_feature_items=weight_feature_m1, weight_feature_users=weight_feature_m2,
        p1=0, p2=0,
        w1=w1, w2=w2,
        normalization=False,
        l1=0, l2=0,
        t1=0, t2=0,
        c1=0, c2=0,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory)

@cython.boundscheck(False)
@cython.wraparound(False)
def popularity_weight_similarity(
    m1, m2,
    weight_pop_m1='sum' , weight_pop_m2='sum',
    float p1=1, float p2=1,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    return s_plus(
        m1, m2,
        weight_pop_items=weight_pop_m1 , weight_pop_users=weight_pop_m2,
        weight_feature_items='none', weight_feature_users='none',
        p1=p1, p2=p2,
        w1=0, w2=0,
        normalization=False,
        l1=0, l2=0,
        t1=0, t2=0,
        c1=0, c2=0,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory)

@cython.boundscheck(False)
@cython.wraparound(False)
def popularity_feature_weight_similarity(
    m1, m2,
    weight_pop_m1='sum' , weight_pop_m2='sum',
    weight_feature_m1='sum' , weight_feature_m2='sum',
    float p1=1, float p2=1,
    float w1=1, float w2=1,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    return s_plus(
        m1, m2,
        weight_pop_items=weight_pop_m1 , weight_pop_users=weight_pop_m2,
        weight_feature_items=weight_feature_m1, weight_feature_users=weight_feature_m2,
        p1=p1, p2=p2,
        w1=w1, w2=w2,
        normalization=False,
        l1=0, l2=0,
        t1=0, t2=0,
        c1=0, c2=0,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory)

@cython.boundscheck(False)
@cython.wraparound(False)
def s_plus_similarity(
    m1, m2,
    weight_pop_m1='none' , weight_pop_m2='none',
    weight_feature_m1='none' , weight_feature_m2='none',
    p1=0, p2=0,
    w1=0, w2=0,
    normalization=True,
    l=0.5,
    c=0.5,
    t1=1, t2=1,
    unsigned int k=100, float shrink=0, float threshold=0,
    binary=False,
    target_items=None,
    verbose = True,
    save_memory = False
    ):
    return s_plus(
        m1, m2,
        weight_pop_items=weight_pop_m1 , weight_pop_users=weight_pop_m2,
        weight_feature_items=weight_feature_m1, weight_feature_users=weight_feature_m2,
        p1=p1, p2=p2,
        w1=w1, w2=w2,
        normalization=normalization,
        l1=l, l2=1-l,
        t1=t1, t2=t2,
        c1=c, c2=1-c,
        k=k, shrink=shrink, threshold=threshold,
        binary=binary,
        target_items=target_items,
        verbose=verbose,
        save_memory=save_memory)
