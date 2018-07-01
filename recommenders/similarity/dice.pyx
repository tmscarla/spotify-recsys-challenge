"""
    author: Simone Boglio
    mail: simone.boglio@mail.polimi.it
"""

import cython
import numpy as np
import scipy.sparse as sp

from cython.operator import dereference
from cython.parallel import parallel, prange
from cython import float

from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef extern from "dice.h" namespace "similarity" nogil:
    cdef cppclass TopK[Index, Value]:
        TopK(size_t K)
        vector[pair[Value, Index]] results

    cdef cppclass SparseMatrixMultiplier[Index, Value]:
        SparseMatrixMultiplier(Index item_count, Value * denColArray, Value shrink, Value threshold)
        void add(Index index, Value value)
        void setDenRow(Value value)
        void foreach[Function](Function & f)


@cython.boundscheck(False)
@cython.wraparound(False)
def dice_similarity(items, unsigned int k=100, float shrink=0, float threshold=0, binary = True, int num_threads=0):
    """ Returns the top K nearest neighbours for each row in the matrix.
    """    
    items.eliminate_zeros() #important, let it
    items = items.tocsr()
    users = items.T.tocsr()
    assert(items.shape[0]==users.shape[1])

    cdef int item_count = items.shape[0]
    cdef int i, u, index1, index2
    cdef float w1

    cdef int[:] item_indptr = items.indptr, item_indices = items.indices
    
    cdef int[:] user_indptr = users.indptr, user_indices = users.indices
    
    
    # build the data terms 
    if binary:
        # save the data (dice use set theorhy)
        old_items = items.data
        old_users = users.data
        items.data = np.ones(items.data.shape[0])
        users.data = np.ones(users.data.shape[0])
        
    cdef float[:] item_data = np.array(items.data, dtype=np.float32)
    cdef float[:] user_data = np.array(users.data, dtype=np.float32)
    
    #build normalization terms 
    cdef float[:] det_item = np.array(items.power(2).sum(axis = 1).A1, dtype=np.float32)
    cdef float[:] det_user = np.array(users.power(2).sum(axis = 0).A1, dtype=np.float32)
    
    if binary:
        # restore data
        items.data = old_items 
        users.data = old_users
    
    # structures for multiplications
    cdef SparseMatrixMultiplier[int, float] * neighbours
    cdef TopK[int, float] * topk
    cdef pair[float, int] result

    # holds triples of output
    cdef float[:] values = np.zeros(item_count * k, dtype=np.float32)
    cdef int[:] rows = np.zeros(item_count * k, dtype=np.int32)
    cdef int[:] cols = np.zeros(item_count * k, dtype=np.int32) 
    
    
    with nogil, parallel(num_threads=num_threads):
        # allocate memory per thread
        neighbours = new SparseMatrixMultiplier[int, float](item_count, &det_user[0], shrink, threshold)
        topk = new TopK[int, float](k)
        try:
            for i in prange(item_count, schedule='dynamic', chunksize=250):
                neighbours.setDenRow(det_item[i])
                for index1 in range(item_indptr[i], item_indptr[i+1]):
                    u = item_indices[index1]
                    w1 = item_data[index1]
                    for index2 in range(user_indptr[u], user_indptr[u+1]):
                        neighbours.add(user_indices[index2],user_data[index2] * w1)
                topk.results.clear()
                neighbours.foreach(dereference(topk))
                index2 = k * i
                for result in topk.results:
                    rows[index2] = i
                    cols[index2] = result.second
                    values[index2] = result.first
                    index2 = index2 + 1

        finally:
            del neighbours
            del topk

    return sp.coo_matrix((values, (rows, cols)),shape=(item_count, item_count), dtype=np.float32)