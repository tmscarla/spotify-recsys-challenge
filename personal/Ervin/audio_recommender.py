import time
import numpy as np
import scipy.sparse as sps
from recommenders.recommender import Recommender
from recommenders.similarity.s_plus import dot_product
from utils.pre_processing import norm_l1_row
from utils.post_processing import eurm_to_recommendation_list
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from personal.Ervin.other_similarity import audio_features_similarity

class Audio_Recommender(Recommender):
    def __init__(self):
        super()

    def fit(self, urm, icm, pid):
        self.urm = urm
        self.pid = pid
        self.icm = icm

    def compute_model(self, knn=100, verbose=False, num_threads=4, save_model=False):
        if verbose:
            print('[ Computing model ]')
            start_time = time.time()

        self.model = audio_features_similarity(self.icm, knn=knn, verbose=verbose, num_threads=num_threads)
        self.model.eliminate_zeros()

        if save_model:
            sps.save_npz('audio_feat_sim_'+str(knn), self.model)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

        return self.model

    def compute_rating(self, topk=750, verbose=False, small=False, mode="offline"):
        if small:
            self.small_urm = self.urm[self.pid]
            self.small_urm = sps.csr_matrix(self.small_urm)
            self.small_urm.eliminate_zeros()

        if verbose:
            print('[ Computing Ratings ]')
            start_time = time.time()

        self.eurm = dot_product(self.small_urm, self.model.T, verbose=verbose, k=topk)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

        return self.eurm.tocsr()

if __name__ == '__main__':
    dr = Datareader(only_load=True, mode='offline', test_num='1', verbose=True)
    pid = dr.get_test_pids()
    urm = dr.get_urm(binary=True)
    ev = Evaluator(dr)

    print('[ Loading ICM with features ]')
    icm = sps.load_npz('/home/ubuntu/Spotify-Challenge/personal/Ervin/audio_features_csr.npz')
    print('[ Done ]')

    rec = Audio_Recommender()
    rec.fit(urm=urm, pid=pid, icm=icm)

    rec.compute_model(knn=100, num_threads=64, verbose=True, save_model=True)
    # rec.model = sps.load_npz('audio_feat_sim_100.npz')
    # print(rec.model.data[rec.model.indptr[22144]:rec.model.indptr[22144]])
    # exit()
    rec.compute_rating(topk=750, verbose=True, small=True)
    ev.evaluate(recommendation_list=eurm_to_recommendation_list(rec.eurm, remove_seed=True, datareader=dr),
                name="AudioRec", old_mode=False)