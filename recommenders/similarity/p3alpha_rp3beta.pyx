"""
    author: Simone Boglio
    mail: simone.boglio@mail.polimi.it
"""

import cython
import numpy as np
import scipy.sparse as sp
import tqdm


from cython.operator import dereference
from cython.parallel import parallel, prange
from cython import address
from cython import float

from scipy.sparse.sputils import get_index_dtype
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp cimport bool

cdef extern from "p3alpha_rp3beta.h" namespace "similarity" nogil:
    cdef cppclass TopK[Index, Value]:
        TopK(size_t K)
        vector[pair[Value, Index]] results

    cdef cppclass SparseMatrixMultiplier[Index, Value]:
        SparseMatrixMultiplier(Index user_count,
                               Value * det_pop,
                               Value alpha,
                               Value mode,
                               Value shrink, Value threshold)
        void add(Index index, Value value)
        void setRow(Index index)
        void foreach[Function](Function & f)
    
cdef extern from "coo.h" nogil:
    void coo_tocsr64(int n_row,int n_col,long nnz,int Ai[],int Aj[],float Ax[],long Bp[],long Bj[],float Bx[])
    void coo_tocsr32(int n_row,int n_col,int nnz,int Ai[],int Aj[],float Ax[],int Bp[],int Bj[],float Bx[])


@cython.boundscheck(False)
@cython.wraparound(False)
def p3alpha_rp3beta_similarity(
    p_iu, p_ui, popularities,
    float alpha=1, float beta=0,
    unsigned int k=100, float shrink=0, float threshold=0,
    target_items=None,
    verbose = True,
    save_memory = False,
    int mode=0,
    int num_threads=0):  

    # change matrix name
    items=p_iu
    users=p_ui

    assert(items.shape[1]==users.shape[0])
    assert((len(popularities)==items.shape[0] and mode==0) or (len(popularities)==users.shape[1] and mode==1))
    assert(target_items is None or len(target_items)<=items.shape[0])
    assert(verbose==True or verbose==False)
    assert(save_memory==True or save_memory==False)
    assert(mode==1 or mode==0)

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

    # build the data terms 
    cdef float[:] item_data = np.array(items.data, dtype=np.float32)
    cdef float[:] user_data = np.array(users.data, dtype=np.float32)

    # build indices and indptrs
    cdef int[:] item_indptr = np.array(items.indptr, dtype=np.int32), item_indices = np.array(items.indices, dtype=np.int32)
    cdef int[:] user_indptr = np.array(users.indptr, dtype=np.int32), user_indices = np.array(users.indices, dtype=np.int32)

    # build normalization terms  for tversky and cosine
    cdef float[:] det_pop = np.power(popularities, beta, dtype=np.float32)

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
        neighbours = new SparseMatrixMultiplier[int, float](user_count, &det_pop[0], alpha, mode, shrink, threshold)
        topk = new TopK[int, float](k)
        try:
            for i in prange(n_targets, schedule='dynamic', chunksize=250):
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
    del det_pop
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
def p3alpha_similarity(p_iu, p_ui, float alpha=1, unsigned int k=100, float shrink=0, float threshold=0, target_items=None, verbose=True, save_memory=False):
    popularities = np.ones(p_iu.shape[0], dtype=np.float32)
    return p3alpha_rp3beta_similarity(
        p_iu, p_ui, popularities,
        alpha=alpha, beta=0,
        k=k, shrink=shrink, threshold=threshold,
        target_items=target_items,
        verbose = verbose,
        save_memory = save_memory,
        mode=0)

@cython.boundscheck(False)
@cython.wraparound(False)
def rp3beta_similarity(p_iu, p_ui, popularities, float beta=1, unsigned int k=100, float shrink=0, float threshold=0, target_items=None, verbose=True, save_memory=False, mode=0):
    return p3alpha_rp3beta_similarity(
        p_iu, p_ui, popularities,
        alpha=1, beta=beta,
        k=k, shrink=shrink, threshold=threshold,
        target_items=target_items,
        verbose = verbose,
        save_memory = save_memory,
        mode=mode)