from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.post_processing import eurm_remove_seed
from utils.post_processing import eurm_to_recommendation_list
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm
from utils.definitions import *
from utils.post_processing import eurm_remove_seed, append_rec_list


dr = Datareader(verbose=False, mode='offline', only_load=True)

urm = dr.get_urm()
urm_col = sps.csc_matrix(urm)
top_p = np.zeros(urm.shape[1])
rec = []
eurm1 = sps.lil_matrix(urm.shape)
eurm2 = sps.lil_matrix(urm.shape)
print(eurm1.shape)
pids = dr.get_test_pids(cat=2)
pids_all = dr.get_test_pids()

# TopPop Album
# ucm_album = dr.get_ucm_albums().tocsc()
# album_dic = dr.get_track_to_album_dict()

# TopPop Artist
ucm_album = dr.get_ucm_albums().tocsc()
artists_dic = dr.get_track_to_artist_dict()


album_to_tracks = load_obj(name="album_tracks_dict_offline",path=ROOT_DIR+"/boosts/")
tracks_to_album = load_obj(name="artist_tracks_dict_offline",path=ROOT_DIR+"/boosts/")



for row in tqdm(pids,desc="part1"):
    track_ind = urm.indices[urm.indptr[row]:urm.indptr[row+1]][0]

    # TopPop Album
    album = artists_dic[track_ind]
    playlists = ucm_album.indices[ucm_album.indptr[album]:ucm_album.indptr[album+1]]

    top = urm[playlists].sum(axis=0).A1.astype(np.int32)



    track_ind_rec = list(top.argsort()[-501:][::-1])

    mask = np.argwhere( tracks_to_album[track_ind_rec]!= album )

    track_ind_rec = track_ind_rec[mask]

    eurm1[row, track_ind_rec] = top[track_ind_rec]

for row in tqdm(pids, desc="part2"):
    track_ind = urm.indices[urm.indptr[row]:urm.indptr[row + 1]][0]

    # TopPop Album
    album = artists_dic[track_ind]
    playlists = ucm_album.indices[ucm_album.indptr[album]:ucm_album.indptr[album + 1]]

    top = urm[playlists].sum(axis=0).A1.astype(np.int32)

    track_ind_rec = top.argsort()[-501:][::-1]

    eurm2[row, track_ind_rec] = top[track_ind_rec]


eurm1 = eurm1.tocsr()[pids_all]
eurm2 = eurm2.tocsr()[pids_all]

eurm1 = eurm_remove_seed(eurm1, dr)
eurm2 = eurm_remove_seed(eurm2, dr)

sps.save_npz("test1.npz", eurm1)

rec_list1=eurm_to_recommendation_list(eurm1)
rec_list2=eurm_to_recommendation_list(eurm2)
rec_list3=append_rec_list(rec_list1+rec_list2)

ev = Evaluator(dr)
ev.evaluate(rec_list1, name="enstest", level='track')
ev.evaluate(rec_list2, name="enstest", level='track')
ev.evaluate(rec_list3, name="enstest", level='track')

    # rec.append(list(top_p))


