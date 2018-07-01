import time
from tqdm import tqdm
import numpy as np
import scipy.sparse as sps
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.post_processing import rec_list_to_eurm,eurm_to_recommendation_list,eurm_remove_seed
import sys

mode = sys.argv[1]

def not_empty_lines_by_target(urm, target_list, min_songs_in_common):
    tmp = urm.tocsc(copy=True)
    tmp = tmp[:, target_list]
    tmp = tmp.tocsr()
    not_empty_lines = list()
    for i in tqdm(range(tmp.shape[0]), desc="eliminating cols"):
        if len(tmp.indices[tmp.indptr[i]:tmp.indptr[i + 1]]) >= min_songs_in_common :
            not_empty_lines.append(i)
    return not_empty_lines


if __name__ == '__main__':

    dr = Datareader(mode=mode, verbose=False, only_load=True)
    ev = Evaluator(datareader=dr)

    urm_csr = dr.get_urm().tocsr()
    urm_csc = sps.csc_matrix(urm_csr.copy())
    # csc_urm_csr.tocsc(copy=True)

    urm_csr.data = np.ones(len(urm_csr.data))
    urm_csc.data = np.ones(len(urm_csc.data))

    test_playlists = dr.get_test_pids(cat=2)
    # test_playlists.extend(dr.get_test_pids(cat=2))

    rec_list = [ [] for x in range(10000)]


    i=1000
    for playlist_id in tqdm(test_playlists, desc="shao belo"):

        songs = urm_csr.indices[urm_csr.indptr[playlist_id]:urm_csr.indptr[playlist_id+1]]

        playlists_with_tokens = urm_csc.indices[ urm_csc.indptr[songs[0]]:urm_csc.indptr[songs[0] + 1]]

        track_total_interactions = urm_csr[playlists_with_tokens].sum(axis=0).A1

        top_pop = track_total_interactions.argsort()[-601:][::-1]

        rec_list[i]= top_pop

        i+=1

    eurm = eurm_remove_seed(rec_list_to_eurm(rec_list) , dr )

    rec_list = eurm_to_recommendation_list(eurm)

    ev.evaluate(rec_list, "cat2_top",verbose=True, do_plot=True, show_plot=True, save=True, )

    sps.save_npz("top_pop_cat2_"+mode, eurm)