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

cdef extern from "dot_product.h" namespace "similarity" nogil:
    cdef cppclass TopK[Index, Value]:
        TopK(size_t K)
        vector[pair[Value, Index]] results

    cdef cppclass SparseMatrixMultiplier[Index, Value]:
        SparseMatrixMultiplier(Index item_count)
        void add(Index index, Value value)
        void foreach[Function](Function & f)

@cython.boundscheck(False)
@cython.wraparound(False)
def dot_product_similarity(items, unsigned int k=100, int num_threads=0):
    return dot_product(items, items.T, k)


@cython.boundscheck(False)
@cython.wraparound(False)
def dot_product(a, b, unsigned int k=100, int num_threads=0):
    """ Returns the top K nearest neighbours for each row in the matrix.
    """
    
    a.eliminate_zeros()
    b.eliminate_zeros()
    items = a.tocsr()
    users = b.tocsr()
    
    assert(items.shape[1]==users.shape[0])

    cdef int item_count = items.shape[0]
    cdef int user_count = users.shape[1]
    cdef int i, u, index1, index2
    cdef float w1

    cdef int[:] item_indptr = items.indptr, item_indices = items.indices
    cdef float[:] item_data = np.array(items.data, dtype=np.float32)

    cdef int[:] user_indptr = users.indptr, user_indices = users.indices
    cdef float[:] user_data = np.array(users.data, dtype=np.float32)

    cdef SparseMatrixMultiplier[int, float] * neighbours
    cdef TopK[int, float] * topk
    cdef pair[float,int] result

    # holds triples of output
    cdef float[:] values = np.zeros(item_count * k, dtype=np.float32)
    cdef int[:] rows = np.zeros(item_count * k, dtype=np.int32)
    cdef int[:] cols = np.zeros(item_count * k, dtype=np.int32)

    with nogil, parallel(num_threads=num_threads):
        # allocate memory per thread
        neighbours = new SparseMatrixMultiplier[int, float](user_count)
        topk = new TopK[int, float](k)
        
        try:
            for i in prange(item_count, schedule='dynamic', chunksize=250):
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

    return sp.coo_matrix((values, (rows, cols)),shape=(item_count, user_count),dtype=np.float32)
    