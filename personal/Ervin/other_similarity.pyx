'''
@author Ervin Dervishaj
@mail vindervishaj@gmail.com
'''

import cython
import numpy as np
import scipy.sparse as sps
import tqdm

from cython import address
from cython.parallel import parallel, prange
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cython import float

cdef extern from "s_plus.h" namespace "similarity" nogil:
    cdef cppclass TopK[Index, Value]:
        TopK(size_t K)
        vector[pair[Value, Index]] results
        void add(Index, Value)


cdef extern from "coo.h" nogil:
    void coo_tocsr64(int n_row,int n_col,long nnz,int Ai[],int Aj[],float Ax[],long Bp[],long Bj[],float Bx[])
    void coo_tocsr32(int n_row,int n_col,int nnz,int Ai[],int Aj[],float Ax[],int Bp[],int Bj[],float Bx[])

@cython.boundscheck(False)
@cython.wraparound(False)
def position_similarity(model, position_urm, knn, verbose=False, num_threads=64):
    """
    :param model: n_items x n_items
    :param position_urm: n_items x n_users (transposed position urm)
    :param knn: knn used to compute model
    :param verbose:
    :return: new model with each similarity divided by norm-1 of the positions of the tracks in different pls
    """

    cdef int n_items, i, j, k, l, idx, users_common, user_i, user_j, out_idx, pos_i, pos_j
    cdef int[:] model_indptr, model_indices, pos_urm_indptr, pos_urm_indices, indices_sim_items, pos_urm_data, \
        indices_user_i, indices_user_j, out_rows, out_cols, data_pos_i, data_pos_j
    cdef float[:] model_data, out_data, data_sim_items
    cdef float avg_dist, sum_distance

    n_items = position_urm.shape[0]

    model_indptr = np.array(model.indptr, dtype=np.int32)
    model_indices = np.array(model.indices, dtype=np.int32)
    model_data = np.array(model.data, dtype=np.float32)

    pos_urm_indptr = np.array(position_urm.indptr, dtype=np.int32)
    pos_urm_indices = np.array(position_urm.indices, dtype=np.int32)
    pos_urm_data = np.array(position_urm.data, dtype=np.int32)

    out_rows = np.zeros(n_items*knn, dtype=np.int32)
    out_cols = np.zeros(n_items*knn, dtype=np.int32)
    out_data = np.zeros(n_items*knn, dtype=np.float32)

    progress = tqdm.tqdm(total=n_items, disable=not verbose)
    progress.desc = 'Updating similarities'
    progress.refresh()

    cdef int n_threads = num_threads
    cdef int verb
    if verbose:
        verb = 1
    else: verb = 0

    cdef int counter = 0
    cdef int * counter_add = address(counter)

    for i in prange(n_items, schedule='dynamic', chunksize=100, nogil=True, num_threads=n_threads):
        # Update progress bar
        if verb:
            counter_add[0] = counter_add[0] + 1
            if counter_add[0]%(n_items/500)==0:
                with gil:
                    progress.desc = 'Updating similarities'
                    progress.n = counter_add[0]
                    progress.refresh()

        # If no similar track to i found continue
        if len(model_indices[model_indptr[i]:model_indptr[i+1]]) == 0:
            continue

        # Index used to iterate over output arrays
        out_idx = model_indptr[i]

        # Index used to iterate over similar track to i
        idx = model_indptr[i]

        # Iterate over all similar tracks to i
        while idx < model_indptr[i+1]:
            # Col of item j similar to i
            j = model_indices[idx]

            # Potential error, there is non-zero similarity of i with itself
            if j == i:
                out_rows[out_idx] = i
                out_cols[out_idx] = j
                out_data[out_idx] = model_data[idx]
                out_idx = out_idx + 1
                idx = idx + 1
                continue

            users_common = 0
            sum_distance = 0

            # Index used to iterate over the users that have i in their playlist
            k = pos_urm_indptr[i]

            # Index used to iterate over the users that have j in their playlist
            l = pos_urm_indptr[j]

            # Col of user that has track j
            user_j = pos_urm_indices[l]

            # Position of track j in playlist user_j
            pos_j = pos_urm_data[l]

            # Iterate over all users that have i in their playlist
            while k < pos_urm_indptr[i+1]:
                # Col of user that has track i
                user_i = pos_urm_indices[k]

                # Position of track i in playlist user_i
                pos_i = pos_urm_data[k]

                # Users (indices) are ordered
                # Iterate over all users that have j in their playlist
                while user_j < user_i and l < pos_urm_indptr[j+1]-1:
                    l = l + 1
                    user_j = pos_urm_indices[l]
                    pos_j = pos_urm_data[l]

                # If same user that has tracks i and j found
                if user_i == user_j:
                    if pos_i - pos_j >= 0:
                        sum_distance = sum_distance + pos_i - pos_j
                    else:
                        sum_distance = sum_distance + pos_j - pos_i
                    # sum_distance = sum_distance + (pos_j - pos_i) ** 2
                    users_common = users_common + 1

                k = k + 1

            # Update the similarity value in model
            avg_dist = sum_distance / users_common
            out_rows[out_idx] = i
            out_cols[out_idx] = j
            out_data[out_idx] = model_data[idx] / avg_dist
            out_idx = out_idx + 1


            idx = idx + 1

    progress.n = n_items
    progress.desc = 'Build csr matrix'
    progress.refresh()

    out_rows = np.trim_zeros(out_rows, 'b')
    out_cols = np.trim_zeros(out_cols, 'b')
    out_data = np.trim_zeros(out_data, 'b')

    coo = sps.coo_matrix((out_data, (out_rows, out_cols)), shape=(n_items, n_items))

    cdef float [:] data
    cdef int [:] indices32, indptr32
    
    cdef int length = len(model_data)

    indptr32 = np.empty(n_items + 1, dtype=np.int32)
    indices32 = np.empty(length, dtype=np.int32)
    data = np.empty(length, dtype=np.float32)
    coo_tocsr32(n_items, n_items, length, &out_rows[0], &out_cols[0], &out_data[0], &indptr32[0], &indices32[0],
               &data[0])

    del out_rows, out_cols, out_data
    
    res = sps.csr_matrix((data, indices32, indptr32), shape=(n_items, n_items), dtype=np.float32)

    del indices32, indptr32

    progress.desc = 'Removing zeros'
    progress.refresh()
    res.eliminate_zeros()

    progress.desc = 'Done'
    progress.refresh()

    return res



@cython.boundscheck(False)
@cython.wraparound(False)
def audio_features_similarity(icm, knn=100, verbose=False, num_threads=64):
    """
    :param icm: n_items x n_features
    :param knn: knn value
    :param verbose
    :param num_threads: threads to be used
    :return: model: n_items x n_items
    """

    cdef int n_items = icm.shape[0]
    cdef int n_features = icm.shape[1]
    cdef int n_threads = num_threads
    cdef int k = knn
    cdef int verb, out_idx, i, j, idx, idx2, idx3, most_s_idx

    cdef int[:] icm_indptr, icm_indices, out_rows, out_cols
    cdef float[:] icm_data, icmT_data, out_data, most_sim
    cdef float sum_diff_feat, feat_i, feat_j

    cdef TopK[int, float] * topk
    cdef pair[float, int] result

    if verbose: verb = 1
    else: verb = 0

    icm_indptr = np.array(icm.indptr, dtype=np.int32)
    icm_indices = np.array(icm.indices, dtype=np.int32)
    icm_data = np.array(icm.data, dtype=np.float32)

    # most_sim = np.zeros(n_items, dtype=np.float)
    # knn_sim = np.zeros(n_items, dtype=np.float)
    # knn_indices = np.zeros(n_items, dtype=np.int32)

    out_rows = np.zeros(n_items*k, dtype=np.int32)
    out_cols = np.zeros(n_items*k, dtype=np.int32)
    out_data = np.zeros(n_items*k, dtype=np.float32)

    progress = tqdm.tqdm(total=n_items, disable=not verbose)
    progress.desc = 'Updating similarities'
    progress.refresh()

    cdef int counter = 0
    cdef int * counter_add = address(counter)

    with nogil, parallel(num_threads=n_threads):
        topk = new TopK[int, float](k)

        for i in prange(n_items, schedule='dynamic', chunksize=100):
            # Update progress bar
            if verb:
                counter_add[0] = counter_add[0] + 1
                # if counter_add[0]%(n_items/500)==0:
                with gil:
                    progress.desc = 'Computing similarities'
                    progress.n = counter_add[0]
                    progress.refresh()

            out_idx = k * i
            most_s_idx = 0

            topk.results.clear()

            # Go over icmT
            for j in range(n_items):
                idx = icm_indptr[i]
                idx2 = icm_indptr[j]

                sum_diff_feat = 0

                # Go over the features
                while idx < icm_indptr[i+1]-1:
                    feat_i = icm_data[idx]
                    feat_j = icm_data[idx2]
                    sum_diff_feat = sum_diff_feat + (feat_i - feat_j) ** 2
                    idx = idx + 1
                    idx2 = idx2 + 1

                if sum_diff_feat == 0:
                    topk.add(most_s_idx, 0)
                else:
                    topk.add(most_s_idx, 1.0 / sum_diff_feat)
                most_s_idx = most_s_idx + 1

            for result in topk.results:
                out_rows[out_idx] = i
                out_cols[out_idx] = result.second
                out_data[out_idx] = result.first
                out_idx = out_idx + 1

    progress.n = n_items
    progress.desc = 'Done'
    progress.refresh()

    coo = sps.coo_matrix((out_data, (out_rows, out_cols)), shape=(n_items, n_items))

    cdef float [:] data
    cdef int [:] indices32, indptr32

    cdef int length = n_items*k

    indptr32 = np.empty(n_items + 1, dtype=np.int32)
    indices32 = np.empty(length, dtype=np.int32)
    data = np.empty(length, dtype=np.float32)

    coo_tocsr32(n_items, n_items, length, &out_rows[0], &out_cols[0], &out_data[0], &indptr32[0], &indices32[0],
               &data[0])

    del out_rows, out_cols, out_data

    res = sps.csr_matrix((data, indices32, indptr32), shape=(n_items, n_items), dtype=np.float32)

    del indices32, indptr32

    return res