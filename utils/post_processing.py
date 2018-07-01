from tqdm import tqdm
import numpy as np
import scipy.sparse as sps
import pandas as pd
from utils.definitions import *
from utils.datareader import Datareader
from utils.pre_processing import norm_max_row
import math
import warnings


def eurm_to_recommendation_list(eurm, cat='all', remove_seed=True, datareader=None, verbose=True):
    """
    Convert the eurm = (10.000, 2M) into a recommendation list if cat is set to 'all', otherwhise
    Convert the eurm = (10.000, 2M) into a recommendation list if a category is specified. #TODO @seba 1k o 10k?
    :param eurm: the estimated user rating matrix
    :param remove_seed: remove seed tracks from playlists
    :param datareader: a Datareader object for seeds removing
    :param cat: 'all' or a value between 1 and 10
    :return: recommendation_list: a list of list of recommendations of shape (10k,500)
    """

    # Convert eurm
    eurm = eurm.tocsr()

    # Remove seeds
    if datareader is None and remove_seed is True and verbose:
        print('[ WARNING! Datareader is None. It was not possible to remove seeds while converting the eurm ]')
    elif datareader is not None and remove_seed is True:
        eurm = eurm_remove_seed(eurm, datareader) # TODO qui crasha se gli passi una matrice da 1k e il remove a True
        if verbose:
            print('Seeds removed!')

    assert cat in ['all', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Initialize rec_list
    recommendation_list = [[] for x in range(10000)]

    if cat == 'all' and eurm.shape[0] == 10000:
        for row in tqdm((range(eurm.shape[0])), desc='Converting eurm', disable= not verbose):
            val = eurm.data[eurm.indptr[row]:eurm.indptr[row+1]]
            ind = val.argsort()[-500:][::-1]
            ind = list(eurm[row].indices[ind])

            recommendation_list[row] = ind

    elif eurm.shape[0] == 1000:
        for row in tqdm(range(eurm.shape[0]), desc='Converting eurm', disable= not verbose):
            val = eurm.data[eurm.indptr[row]:eurm.indptr[row+1]]
            ind = val.argsort()[-500:][::-1]
            ind = list(eurm[row].indices[ind])
            recommendation_list[row + (cat - 1) * 1000] = ind
    else:
        raise Exception("Configuration of cat parameter and urm shape not correct")

    return recommendation_list


def adaptive_eurm_to_recommendation_list(eurm, verbose=True):
    """
    Returns a recommendation list as long as the urm provided.
    An urm (10k,2kk) will result in a (10k,500) rec list, a (3k,2kk) will get (3k,500) recommendations
    :param eurm: estimated user rating matrix
    :return: list of lists of recommendations.
    """
    eurm = eurm.tocsr()

    recommendation_list = [[] for x in range(eurm.shape[0])]

    for row in tqdm(range(eurm.shape[0]), desc='Converting eurm (adapt)', disable=not verbose):
        val = eurm.data[eurm.indptr[row]:eurm.indptr[row + 1]]
        ind = val.argsort()[-500:][::-1]
        ind = list(eurm[row].indices[ind])
        recommendation_list[row] = ind

    return recommendation_list


def eurm_to_recommendation_list_submission(eurm, remove_seed=True, datareader=None, verbose=True):
    """
    Convert the eurm = (10.000, 2.2M) into a recommendation list, deleting all the value <= 0 from the eurm
    :param eurm: the estimated user rating matrix
    :param remove_seed: remove seed tracks
    :param datareader: a Datareader object for removing seed tracks
    :return: recommendation_list: a list of list of recommendations
    """
    # Convert eurm
    eurm = eurm.tocsr()

    # Remove seeds
    if datareader is None and remove_seed is True:
        print('[ WARNING! Datareader is None. It was not possible to remove seeds while converting the eurm ]')
    elif datareader is not None and remove_seed is True:
        eurm = eurm_remove_seed(eurm, datareader)
        if verbose:
            print('Seeds removed!')

    # Initialize rec_list
    recommendation_list = [[] for x in range(10000)]

    # Remove <= 0 values from eurm, now .data has only positive values
    eurm.data[eurm.data <= 0] = 0
    eurm.eliminate_zeros()

    for row in tqdm((range(eurm.shape[0])), desc='Converting eurm for submission', disable=not verbose):
        val = eurm.data[eurm.indptr[row]:eurm.indptr[row + 1]]
        ind = val.argsort()[-500:][::-1]
        ind = list(eurm[row].indices[ind])

        if len(ind) < 500:
            print('ATTENTION: found', len(ind), '< 500 recommendations in row:', row)

        recommendation_list[row] = ind

    return recommendation_list


def eurm_remove_seed(eurm, datareader, eliminate_negative=True):
    """
    Remove seed tracks from the eurm (10K, 2M)
    :param eurm: original eurm
    :param datareader: a Datareader object, the same used to build the original eurm
    :return: eurm: eurm with no seed tracks
    """
    # Convert eurm
    eurm = eurm.tocsr()

    # Get urm with shape of eurm
    urm = datareader.get_urm()
    pids = datareader.get_test_pids()
    urm_test = urm[pids]
    max_value = eurm.max()

    new_data = np.ones(len(urm_test.data)) * max_value
    urm_test.data = new_data

    # Remove seen
    eurm = eurm - urm_test

    if eliminate_negative:
        eurm.data[eurm.data <= 0] = 0
        eurm.eliminate_zeros()

    return eurm


def rec_list_to_eurm(rec_list, from_500=False):
    """
    :param rec_list: array of arrays of shape  [10k , x], with the indices to recommend.
    :return: EURM of shape (10k,500) with incremental integer values for each row
    """
    data = []
    row = []
    col = []

    for i in range(len(rec_list)):
        if from_500:
            data.extend(np.arange(start=500, stop=0, step=-1, dtype=np.int32))
        else:
            data.extend(np.arange(start=len(rec_list[i]), stop=0, step=-1, dtype=np.int32))

        col.extend((rec_list[i]))
        row.extend(np.ones(len(rec_list[i]),dtype=np.int32)*i)

    eurm = sps.csr_matrix( (data, (row, col)), shape=(len(rec_list), 2262292), dtype=np.int32)

    return eurm


def reorder_old_eurm(eurm):
    """
    ATTENTION: this function is intended to be used only for old eurms, which are
    ordered by test pids and not by categories.
    :param eurm: the old-ordered eurm
    :return: eurm: the new-ordered eurm
    """

    dr_old = Datareader(mode='online', only_load='True', type='old')

    res = []
    for cat in range(1,11):
        indices = dr_old.get_test_pids_indices(cat=cat)
        res.append(eurm[indices])

    eurm_new = sps.vstack(res)

    return eurm_new


def norm_tanh(eurm, from_500=False, hard_curve=False):
    """
    it transforms the rating eurm provided to a ranking based.
    it loses the information of the rating and will split the values on a tanh(x) curve
    :param eurm:        estimated user rating matrix
    :param from_500:    if true it will push the first elements of rec_lists with a few recommendations
                        to have the same weight like the first ones of the full recommendations
    :param hard_curve:  uses only the tanh(x) before the "flesso" point
    :return:
    """
    check_shape = eurm.shape[0]
    rec_list = adaptive_eurm_to_recommendation_list(eurm)
    eurm_new = rec_list_to_eurm(rec_list, from_500=from_500)

    eurm_new = norm_max_row(eurm_new)

    eurm_new.data = eurm_new.data*500

    if hard_curve:
        eurm_new.data = (np.tanh((eurm_new.data - 500) / 250) + 1) /2
    else:
        eurm_new.data = (np.tanh((eurm_new.data - 250) / 125) + 1) /2

    assert check_shape == eurm_new.shape[0]
    return eurm_new


def eurm_remove_tracks_before_cutoff(eurm, datareader):
    """
    For each row of the eurm eliminate the predictions until the cutoff value.
    :param eurm: the eurm to be modified
    :return: eurm: the eurm with the values removed
    """

    # Eventually convert eurm to csr
    eurm = eurm.tocsr()

    # Get array of holdouts
    holdouts = datareader.get_df_test_playlists()['num_holdouts'].as_matrix()

    for row in range(eurm.shape[0]):
        values = eurm.data[eurm.indptr[row]:eurm.indptr[row + 1]]
        cutoff_indices = values.argsort()[-holdouts[row]:][::-1]
        values[cutoff_indices] = 0

        eurm.data[eurm.indptr[row]:eurm.indptr[row + 1]] = values

    eurm.eliminate_zeros()

    return eurm


def remove_predictions_from_eurm(eurm_source, eurm_target, datareader, cut_off=True):
    """
    For each row of the target eurm, remove tracks predicted by the source urm.
    :param eurm_source: eurm (10k x 2.2M)
    :param eurm_target: eurm (10k x 2.2M)
    :param datareader: a Datareader object
    :param cut_off: remove predictions from source eurm only until the cut_off
    :return: eurm_target: the target eurm with values removed
    """

    assert eurm_source.shape == eurm_target.shape

    # Eventually convert eurms to csr
    eurm_source = eurm_source.tocsr()
    eurm_target = eurm_target.tocsr()

    # Get array of holdouts
    holdouts = datareader.get_df_test_playlists()['num_holdouts'].as_matrix()

    for row in tqdm(range(eurm_source.shape[0]), desc='Removing predictions'):

        # Row initialization
        row_start_s = eurm_source.indptr[row]
        row_end_s = eurm_source.indptr[row + 1]
        row_start_t = eurm_target.indptr[row]
        row_end_t = eurm_target.indptr[row + 1]

        # Select indices and data of source and target
        values_s = eurm_source.data[row_start_s:row_end_s]
        indices_s = eurm_source.indices[row_start_s:row_end_s]
        values_t = eurm_target.data[row_start_t:row_end_t]
        indices_t = eurm_target.indices[row_start_t:row_end_t]

        # Find tracks to be removed
        if cut_off:
            internal_indices = np.argsort(values_s)[::-1][:holdouts[row]]
            indices_to_remove = indices_s[internal_indices]
            mask = np.isin(indices_t, indices_to_remove)
        else:
            mask = np.isin(indices_t, indices_s)

        # Set values of found tracks to 0
        values_t[mask] = 0
        eurm_target.data[row_start_t:row_end_t] = values_t

    eurm_target.eliminate_zeros()

    return eurm_target


def append_rec_list(strong_rec_list, weak_rec_list):
    """
    Appends a weak recommendation list to a strong one. It checks if the items of the second are already
    inside the strong one and add the remaining ones
    Example: [1,2,3] + [0,3,2,4] >>> [1,2,3,0,4]
    :param strong_rec_list: rec list to leave in the first positions
    :param weak_rec_list: rec list to leave in the last positions, removing duplicates
    :return: ensembled_rec_list: the ensembled rec_list
    """

    def __remove_duplicates_preserving_order(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    assert len(strong_rec_list) == len(weak_rec_list)

    ensembled_rec_list = []

    for i in range(len(strong_rec_list)):
        new_list = strong_rec_list[i] + weak_rec_list[i]
        ensembled_rec_list.append(__remove_duplicates_preserving_order(new_list)[0:500])

    return ensembled_rec_list


def remove_rec_list_after_cutoff(rec_list, datareader):
    """
    For each recommendation list, remove all the recommendations after the cutoff.
    :param rec_list: a list of recommendation lists
    :param datareader: a Datareader object
    :return: rec_list_shrinked: a rec list with no recommendations after cutoff
    """

    # Get array of holdouts
    holdouts = datareader.get_df_test_playlists()['num_holdouts'].as_matrix()

    # Initialize shrinked rec_list
    rec_list_shrinked = []

    for i in range(len(rec_list)):
        row = rec_list[i]
        holdout = holdouts[i]

        rec_list_shrinked.append(row[:holdout])

    return rec_list_shrinked


def shift_rec_list_cutoff(rec_list, datareader):
    """
    For each recommendation list, shift the recommendation to the bottom for a number of positions = cutoff.
    The values in range [0:cutoff] will be filled with a random track which is not present in the holdouts
    and therefore will not impact the evaluation.
    :param rec_list: a list of recommendation lists to be shifted
    :param datareader: a Datareader object
    :return: rec_list_shifted: the shifted rec_list
    """

    if datareader.online():
        print('ATTENTION: this method is intended to be used only for offline evaluation!!!')

    # Get array of holdouts
    holdouts = datareader.get_df_test_playlists()['num_holdouts'].as_matrix()

    # Initialize shifted rec_list
    rec_list_shifted = []

    # Set as bad track the last track of the dataset which is not present in our test set
    bad_track = 2262291

    for i in tqdm(range(len(rec_list)), desc='Shifting rec_list'):
        # Initialization
        row = rec_list[i]
        holdout = holdouts[i]

        # Create new row
        new_row = [bad_track for x in range(holdout)]
        new_row = new_row + row
        new_row = new_row[:500]

        rec_list_shifted.append(new_row)

    return rec_list_shifted


def shuffle_rec_list_before_cutoff(rec_list, datareader):
    """
    Randomly shuffle the rec_list before the cutoff.
    :param rec_list: a list of recommendation lists to be shuffled
    :param datareader: a Datareader object
    :return: rec_list_shuffled: the shuffled rec_list
    """

    # Get array of holdouts
    holdouts = datareader.get_df_test_playlists()['num_holdouts'].as_matrix()

    # Initialize shifted rec_list
    rec_list_shuffled = []

    for i in tqdm(range(len(rec_list)), desc='Shuffling rec_list'):
        # Initialization
        row = rec_list[i]
        holdout = holdouts[i]

        # Create new row
        new_row = row[:holdout]
        np.random.shuffle(new_row)
        new_row = new_row + row[holdout:]

        rec_list_shuffled.append(new_row)

    return rec_list_shuffled


def combine_two_eurms(first_eurm, second_eurm, cat_first):
    """
    Combine two eurms of both shape (10k, 2.2M) into a new eurm which has the predictions of the first eurm
    for the categories in 'cat_first' and the predictions of the second for the rest of the categories.
    :param first_eurm: first estimated user rating matrix
    :param second_eurm: second estimated user rating matrix
    :param cat_first: a list of categories e.g. [3, 5, 8, 10] for the first eurm
    :return: eurm_combined: the new eurm combined
    """

    rows = []
    cols = []
    data = []

    for i in tqdm(range(first_eurm.shape[0]), desc='Combining eurms'):
        cat_i = int(i / 1000) + 1

        # First matrix
        if cat_i in cat_first:
            start = first_eurm.indptr[i]
            end = first_eurm.indptr[i + 1]

            tracks = first_eurm.indices[start:end]
            values = first_eurm.data[start:end]

        # Second matrix
        else:
            start = second_eurm.indptr[i]
            end = second_eurm.indptr[i + 1]

            tracks = second_eurm.indices[start:end]
            values = second_eurm.data[start:end]

        rows.extend([i for x in range(len(tracks))])
        cols.extend(tracks)
        data.extend(values)

    eurm_combined = sps.csr_matrix((data, (rows, cols)), shape=first_eurm.shape)

    return eurm_combined
