from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.post_processing import eurm_remove_seed
from utils.post_processing import eurm_to_recommendation_list
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm

dr = Datareader(verbose=False, mode='online', only_load=True)

urm = dr.get_urm()
urm_col = sps.csc_matrix(urm)
top_p = np.zeros(urm.shape[1])
rec = []
eurm = sps.lil_matrix(urm.shape)
print(eurm.shape)
pids = dr.get_test_pids(cat=2)
pids_all = dr.get_test_pids()

# TopPop Album
# ucm_album = dr.get_ucm_albums().tocsc()
# album_dic = dr.get_track_to_album_dict()

# TopPop Artist
# ucm_artist = dr.get_ucm_artists().tocsc()
# artists_dic = dr.get_track_to_artist_dict()

for row in tqdm(pids):
    track_ind = urm.indices[urm.indptr[row]:urm.indptr[row+1]][0]

    # # TopPop Artist
    # artists = artists_dic[track_ind]
    # playlists = ucm_artist.indices[ucm_album.indptr[artists]:ucm_album.indptr[artists+1]]

    # TopPop Album
    # album = album_dic[track_ind]
    # playlists = ucm_album.indices[ucm_album.indptr[album]:ucm_album.indptr[album+1]]

    # TopPop Track
    playlists = urm_col.indices[urm_col.indptr[track_ind]:urm_col.indptr[track_ind+1]]

    top = urm[playlists].sum(axis=0).A1.astype(np.int32)
    track_ind_rec = top.argsort()[-501:][::-1]


    eurm[row, track_ind_rec] = top[track_ind_rec]


eurm = eurm.tocsr()[pids_all]
eurm = eurm_remove_seed(eurm, dr)

sps.save_npz("top_pop_2_track_online.npz", eurm)
# ev = Evaluator(dr)
# ev.evaluate(eurm_to_recommendation_list(eurm), name="enstest", level='track')

    # rec.append(list(top_p))


