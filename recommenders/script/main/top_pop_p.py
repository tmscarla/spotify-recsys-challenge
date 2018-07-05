from utils.datareader import Datareader
from utils.post_processing import eurm_remove_seed
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm
import sys

class Top_pop_p(object):
    '''
    Class that allow the user to get the personalized top pop build following track or album
    '''
    def __init__(self):

        self.dr_on = Datareader(verbose=False, mode='online', only_load=True)
        self.dr_of = Datareader(verbose=False, mode='offline', only_load=True)
        self.urm_on = self.dr_on.get_urm()
        self.urm_of = self.dr_of.get_urm()
        self.urm_col = sps.csc_matrix(self.urm_of)
        self.top_p = np.zeros(self.urm_of.shape[1])


    def get_top_pop_album(self, mode):
        '''
        :return: csr_matrix filled with the reccomendation for the cat 2 following album
        '''
        if mode=="online":
            eurm = sps.lil_matrix(self.urm_on.shape)
            pids = self.dr_on.get_test_pids(cat=2)
            pids_all = self.dr_on.get_test_pids()
            ucm_album = self.dr_of.get_ucm_albums().tocsc()
            album_dic = self.dr_of.get_track_to_album_dict()

            for row in tqdm(pids):
                track_ind = self.urm_on.indices[self.urm_on.indptr[row]:self.urm_on.indptr[row + 1]][0]

                album = album_dic[track_ind]
                playlists = ucm_album.indices[ucm_album.indptr[album]:ucm_album.indptr[album+1]]

                top = self.urm_of[playlists].sum(axis=0).A1.astype(np.int32)
                track_ind_rec = top.argsort()[-501:][::-1]

                eurm[row, track_ind_rec] = top[track_ind_rec]

            eurm = eurm.tocsr()[pids_all]
            eurm = eurm_remove_seed(eurm, self.dr_on)

        elif mode=="offline":
            eurm = sps.lil_matrix(self.urm_of.shape)
            pids = self.dr_of.get_test_pids(cat=2)
            pids_all = self.dr_of.get_test_pids()
            ucm_album = self.dr_of.get_ucm_albums().tocsc()
            album_dic = self.dr_of.get_track_to_album_dict()

            for row in tqdm(pids):
                track_ind = self.urm_of.indices[self.urm_of.indptr[row]:self.urm_of.indptr[row + 1]][0]

                album = album_dic[track_ind]
                playlists = ucm_album.indices[ucm_album.indptr[album]:ucm_album.indptr[album+1]]

                top = self.urm_of[playlists].sum(axis=0).A1.astype(np.int32)
                track_ind_rec = top.argsort()[-501:][::-1]

                eurm[row, track_ind_rec] = top[track_ind_rec]

            eurm = eurm.tocsr()[pids_all]
            eurm = eurm_remove_seed(eurm, self.dr_of)

        return eurm.copy().tocsr()

    def get_top_pop_track(self, mode):
        '''
        :return: csr_matrix filled with the reccomendation for the cat 2 following track
        '''
        if mode=="online":

            eurm = sps.lil_matrix(self.urm_on.shape)
            pids = self.dr_on.get_test_pids(cat=2)
            pids_all = self.dr_on.get_test_pids()

            for row in tqdm(pids):
                track_ind = self.urm_on.indices[self.urm_on.indptr[row]:self.urm_on.indptr[row + 1]][0]

                playlists =  self.urm_col.indices[ self.urm_col.indptr[track_ind]: self.urm_col.indptr[track_ind+1]]

                top = self.urm_of[playlists].sum(axis=0).A1.astype(np.int32)
                track_ind_rec = top.argsort()[-501:][::-1]

                eurm[row, track_ind_rec] = top[track_ind_rec]

            eurm = eurm.tocsr()[pids_all]
            eurm = eurm_remove_seed(eurm, self.dr_on)

        elif mode=="offline":

            eurm = sps.lil_matrix(self.urm_of.shape)
            pids = self.dr_of.get_test_pids(cat=2)
            pids_all = self.dr_of.get_test_pids()

            for row in tqdm(pids):
                track_ind = self.urm_of.indices[self.urm_of.indptr[row]:self.urm_of.indptr[row + 1]][0]

                playlists = self.urm_col.indices[self.urm_col.indptr[track_ind]: self.urm_col.indptr[track_ind + 1]]

                top = self.urm_of[playlists].sum(axis=0).A1.astype(np.int32)
                track_ind_rec = top.argsort()[-501:][::-1]

                eurm[row, track_ind_rec] = top[track_ind_rec]

            eurm = eurm.tocsr()[pids_all]
            eurm = eurm_remove_seed(eurm, self.dr_of)

        return eurm.copy().tocsr()

if __name__ == '__main__':
    t = Top_pop_p()
    t.get_top_pop_album("online")