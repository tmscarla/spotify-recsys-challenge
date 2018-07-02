"""
@author Ervin Dervishaj
@email vindervishaj@gmail.com
"""

from recommenders.recommender import Recommender
from recommenders.similarity.s_plus import *
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.post_processing import eurm_to_recommendation_list

import time
import numpy as np
from scipy import sparse as sps
from sklearn.preprocessing import normalize


class TF_collaborative_user(Recommender):
    def __init__(self):
        super()

    def compute_model(self, knn=100, power=1.0, verbose=False, save_model=False, target_items=None):
        if verbose:
            print("[ Creating model with user TF-IDF similarity ]")
            start_time = time.time()

        # Calculate DF[t] & IDF[t]
        dft = self.urm.sum(axis=0).A1
        idft = np.log(self.urm.shape[0] / (dft + 1e-8))

        # Multiply each listened track with its respective idf
        URM_enhanced = self.urm.multiply(idft).tocsr()

        # Get the user similarity matrix
        self.model = dot_product(URM_enhanced, self.urm.T, k=knn, verbose=verbose, target_items=target_items)
        self.model = self.model.tolil()
        self.model.setdiag(np.zeros(self.model.shape[0]))
        self.model = self.model.tocsr()
        self.model.eliminate_zeros()
        self.model.data = np.power(self.model.data, power)

        if save_model:
            if verbose:
                print('[ Saving the model ]')
            sps.save_npz('tf_idf_user_sim_'+str(knn), self.model)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

        return self.model

    def compute_rating(self, top_k=500, verbose=False, small=False):
        if small:
            self.model = sps.csr_matrix(self.model.tocsr()[self.pid])
        self.urm = sps.csr_matrix(self.urm)
        self.model = sps.csr_matrix(self.model)

        if verbose:
            print("[ Compute ratings ]")
            start_time = time.time()

        # Normalize the original URM to get cv for each track listened by the users
        user_pen = normalize(self.urm, axis=1, norm='l1')

        # Calculate DF[t] & IDF[t]
        dft = self.urm.sum(axis=0).A1
        idft = np.log(self.urm.shape[0] / (dft + 1e-8))

        # Multiply each listened track with its respective idf
        URM_enhanced = self.urm.multiply(idft).tocsr()

        # Computer the eURM
        self.eurm = dot_product(self.model, user_pen, k=top_k, verbose=verbose, target_items=self.pid)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

        return self.eurm

if __name__ == '__main__':
    dr = Datareader(verbose=True, mode='offline', only_load=True)
    urm = dr.get_urm(binary=True)
    pid = dr.get_test_pids()
    ev = Evaluator(dr)

    topk = 750

    configs = [
        {'cat': 10, 'knn': 100, 'power': 2.4},
        {'cat': 9, 'knn': 200, 'power': 0.4},
        {'cat': 8, 'knn': 100, 'power': 2},
        {'cat': 7, 'knn': 300, 'power': 1},
        {'cat': 6, 'knn': 300, 'power': 2},
        {'cat': 5, 'knn': 500, 'power': 2.4},
        {'cat': 4, 'knn': 300, 'power': 1.8},
        {'cat': 3, 'knn': 200, 'power': 2.2},
        {'cat': 2, 'knn': 500, 'power': 1}
    ]

    eurm = sp.csr_matrix(urm.shape)
    rec = TF_collaborative_user()

    for c in configs:
        pid = dr.get_test_pids(cat=c['cat'])
        rec.fit(urm, pid)
        rec.compute_model(verbose=True, knn=c['knn'], save_model=False, power=c['power'], target_items=pid)
        rec.compute_rating(top_k=topk, verbose=True, small=False)
        eurm = eurm + rec.eurm
        del rec.eurm
        del rec.model

    pids = dr.get_test_pids()
    eurm = eurm[pids]
    ev.evaluate(recommendation_list=eurm_to_recommendation_list(eurm, datareader=dr, remove_seed=True),
                name="tfidf_collaborative_user", old_mode=False)