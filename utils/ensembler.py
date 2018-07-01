from sklearn.preprocessing import normalize
import scipy.sparse as sps
import logging
logging.basicConfig(filename='result.log', level=logging.DEBUG)
from tqdm import tqdm
import numpy as np


def cat_ensembler(matrix_l, weight, cat, normalization_type="none", axis=1):
    """
    :param matrix_l: list of sparse_matrix
    :param weight: list of weight
    :param normalization_type:(string) none, l1, l2
    :param axis: axis where perform the normalization
    :return: ensembled matrix
    """

    assert len(weight) == len(matrix_l)
    for i in matrix_l:
        assert matrix_l[0].shape == i.shape

    if normalization_type != "none":
        assert normalization_type == "l1" or normalization_type == "l2" or normalization_type == "max"
        assert axis == 1 or axis == 0
        for m in range(len(matrix_l)):
            matrix_l[m] = normalize(matrix_l[m], norm=normalization_type, axis=1)
    ens = matrix_l[0]*weight[0]
    for m in range(1, len(matrix_l)):
        ens += matrix_l[m]*weight[m]
    return ens[cat*1000:(cat*1000)+1000]


def ranking_ensembler(eurms_list, weigths=[], top_k=[], penalization=500):
    """
    Ensemble a list of eurms looking at the ranking of each recommendation.
    :param eurms_list: a list of estimated user rating matrices to be ensembled
    :param weigths: a list of weights for each eurm. By default each weight is equal to 1.0.
           Final ranking will be computed as follows:
    :param top_k: a list of top_k tracks to be considered for each row of each eum
    :param penalization: a penalization to apply to the ranking of a track each time
           is not present in a recommender
    :return: eurm_ensembled: the ensembled eurm
    """

    # Check if matrices have all the same shape
    for eurm in eurms_list:
        assert eurms_list[0].shape == eurm.shape

    # Convert all to csr format if necessary
    for eurm in eurms_list:
        eurm = eurm.tocsr()

    # Set default weights if not specified
    if len(weigths) > 0:
        assert len(eurms_list) == len(weigths)
    else:
        weigths = [1.0 for x in range(len(eurms_list))]

    # Set default top_k if not specified
    if len(top_k) > 0:
        assert len(eurms_list) == len(top_k)
    else:
        top_k = [2000 for x in range(len(eurms_list))]

    # Init data structures
    rows = []
    cols = []
    data = []

    for row in tqdm(range(eurms_list[0].shape[0]), desc='Ranking ensemble'):

        # Initialize dict for each row { track: total_rank }
        ranking_dict = dict()

        for i in range(len(eurms_list)):
            eurm = eurms_list[i]

            row_start = eurm.indptr[row]
            row_end = eurm.indptr[row+1]
            row_columns = eurm.indices[row_start:row_end]
            row_data = eurm.data[row_start:row_end]

            top_k_ratings = np.argsort(row_data)[::-1][:top_k[i]]
            top_k_tracks = row_columns[top_k_ratings]

            for rank in range(len(top_k_tracks)):
                track = top_k_tracks[rank]

                if track in ranking_dict.keys():
                    ranking_dict[track][0] += (rank+1) * weigths[i]
                    ranking_dict[track][1] += 1
                else:
                    ranking_dict[track] = [(rank+1) * weigths[i], 1]

        # Fill ensembled eurm
        for track in ranking_dict.keys():
            rows.append(row)
            cols.append(track)
            data.append(ranking_dict[track][0] + (penalization * (len(eurms_list) - ranking_dict[track][1])))

    # Compute element-wise inversion
    data = np.array(data)
    data = 1. / data

    eurm_ensembled = sps.csr_matrix((data, (rows, cols)), shape=eurms_list[0].shape)

    return eurm_ensembled



































def ensembler(matrix, weight, normalization_type="none" ,axis=1):
    '''
    :param matrix: list of sparse_matrix
    :param weight: list of weight
    :param normalization_type:(string) none, l1, l2
    :param axis: axis where perform the normalization
    :return: ensembled matrix
    '''

    assert len(weight)== len(matrix)
    for i in matrix:
        assert matrix[0].shape == i.shape


    if normalization_type != "none":
        assert normalization_type == "l1" or normalization_type == "l2" or normalization_type == "max"
        assert axis == 1 or axis == 0
        for m in range(len(matrix)):
            matrix[m] = normalize(matrix[m], norm=normalization_type, axis=1)

    ens = matrix[0]*weight[0]
    for i in range(len(matrix)-1):
        ens = ens + matrix[i + 1]*weight[i+1]
    return ens




if __name__ == '__main__':
    top_p = sps.load_npz("top_pop.npz")
    print("loaded")
    matrix = [top_p, top_p]
    w = [1,1]

    print("runing")
    res = cat_ensembler(matrix, w, 0)
    print(res.nnz)
    print(top_p[0])
    print(res[0])









