"""
python gen_clustered_matrices_main.py offline

"""
from utils.sparse import inplace_set_rows_zero
from scipy.sparse import csr_matrix
from utils.datareader import Datareader
from utils.definitions import ROOT_DIR
import numpy as np
import scipy.sparse as sps
import os
import sys

mode = sys.argv[1] # online or offline

if __name__ == '__main__':

    dr = Datareader(mode=mode, only_load=True, verbose=False)

    track_to_artist_dict = dr.get_track_to_artist_dict()
    track_to_album_dict = dr.get_track_to_album_dict()

    test_pids = dr.get_test_pids()
    urm = dr.get_urm()
    popularity = np.diff(urm.tocsc().indptr)

    cluster_1_artist = list()
    cluster_2_artist = list()
    cluster_3_artist = list()
    cluster_4_artist = list()

    ## clustering the matrices
    for prog, pid in enumerate(test_pids):
        songs = urm[pid].nonzero()[1]

        if len(songs) > 0:
            artists_array = [track_to_artist_dict[song] for song in songs]

            unique_songs_l = len(np.unique(songs))
            unique_artists_l = len(np.unique(artists_array))

            tracks_div_artists = np.log2(unique_songs_l / unique_artists_l)

            if tracks_div_artists == 0:
                cluster_1_artist.append(prog)
            elif tracks_div_artists < 1:
                cluster_2_artist.append(prog)
            elif tracks_div_artists < 2:
                cluster_3_artist.append(prog)
            else:
                cluster_4_artist.append(prog)


    directory_npz = ROOT_DIR + '/recommenders/script/main/'+mode+'_npz/'

    ## folder creation
    ar1 = ROOT_DIR+'/recommenders/script/main/'+mode+'_npz/npz_ar1/'
    ar2 = ROOT_DIR+'/recommenders/script/main/'+mode+'_npz/npz_ar2/'
    ar3 = ROOT_DIR+'/recommenders/script/main/'+mode+'_npz/npz_ar3/'
    ar4 = ROOT_DIR+'/recommenders/script/main/'+mode+'_npz/npz_ar4/'

    folders = [ar1,ar2,ar3,ar4]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


    filenames = list(
        map(lambda x: directory_npz + x, list(filter(lambda x: mode in x, os.listdir(directory_npz)))))

    ## selecting lines for each cluster
    all_lines = np.array([x for x in range(10000)])

    cluster_1_artist = np.array(cluster_1_artist).ravel()
    cluster_2_artist = np.array(cluster_2_artist).ravel()
    cluster_3_artist = np.array(cluster_3_artist).ravel()
    cluster_4_artist = np.array(cluster_4_artist).ravel()

    cluster_1_artist_skip_lines = np.setdiff1d(all_lines, cluster_1_artist)
    cluster_2_artist_skip_lines = np.setdiff1d(all_lines, cluster_2_artist)
    cluster_3_artist_skip_lines = np.setdiff1d(all_lines, cluster_3_artist)
    cluster_4_artist_skip_lines = np.setdiff1d(all_lines, cluster_4_artist)


    ## writing the clustered matrices

    for path_eurm in filenames:

        eurm = sps.load_npz(path_eurm)
        print(path_eurm)

        eurm_cluster_1_art = csr_matrix(eurm.copy())
        inplace_set_rows_zero(eurm_cluster_1_art, cluster_1_artist_skip_lines)
        sps.save_npz(ar1 + path_eurm.split('/')[-1], eurm_cluster_1_art.tocsr())

        eurm_cluster_2_art = csr_matrix(eurm.copy())
        inplace_set_rows_zero(eurm_cluster_2_art, cluster_2_artist_skip_lines)
        sps.save_npz(ar2 + path_eurm.split('/')[-1], eurm_cluster_2_art.tocsr())

        eurm_cluster_3_art = csr_matrix(eurm.copy())
        inplace_set_rows_zero(eurm_cluster_3_art, cluster_3_artist_skip_lines)
        sps.save_npz(ar3 + path_eurm.split('/')[-1], eurm_cluster_3_art.tocsr())


        eurm_cluster_4_art = csr_matrix(eurm.copy())
        inplace_set_rows_zero(eurm_cluster_4_art, cluster_4_artist_skip_lines)
        sps.save_npz(ar4 + path_eurm.split('/')[-1], eurm_cluster_4_art.tocsr())

