from recommenders.recommender import Recommender
from recommenders.similarity.s_plus import dot_product
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.post_processing import eurm_to_recommendation_list
from personal.Ervin.other_similarity import position_similarity

import time, sys
import numpy as np
from scipy import sparse as sps
from sklearn.preprocessing import normalize


class TF_collaborative_item(Recommender):
    def __init__(self):
        super()

    def compute_model(self, knn=100, verbose=False, power=1, save_model=False):
        if verbose:
            print("[ Creating model with item TF-IDF similarity ]")
            start_time = time.time()

        # Calculate DF[u] & IDF[u]
        urm_bin = sps.csr_matrix(self.urm)
        urm_bin.data = np.ones(len(self.urm.data))
        dft = urm_bin.sum(axis=1).A1
        idft = np.log(self.urm.shape[1] / (dft + 1e-8))

        # dft = self.urm.sum(axis=1).A1
        # idft = np.log(self.urm.shape[1] / (dft + 1e-8))

        # Multiply each listened track with its respective idf
        URM_enhanced = self.urm.multiply(idft.reshape(-1,1)).tocsr()

        # Get the user similarity matrix
        self.model = dot_product(URM_enhanced.T, self.urm, k=knn, verbose=verbose)
        self.model = self.model.tolil()
        self.model.setdiag(np.zeros(self.model.shape[0]))
        self.model = self.model.tocsr()
        self.model.eliminate_zeros()
        self.model.data = np.power(self.model.data, power)

        if save_model:
            if verbose:
                print('[ Saving the model ]')
            sps.save_npz('tf_idf_item_sim_' + str(knn), self.model)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

        return self.model

    def compute_rating(self, top_k=500, verbose=False, small=False):
        if small:
            self.urm = sps.csr_matrix(self.urm[self.pid])
        self.model = sps.csr_matrix(self.model)

        if verbose:
            print("[ Compute ratings ]")
            start_time = time.time()

        # Normalize the original URM to get pop for each track
        norm_urm = normalize(self.urm, axis=0, norm='l1')

        # dft = self.urm.sum(axis=0).A1
        # idft = np.log(self.urm.shape[0] / (dft + 1e-8))
        # idft = np.power(idft, 0.5)
        # norm_urm = self.urm.multiply(idft.reshape(1,-1)).tocsr()

        # Computer the eURM
        self.eurm = dot_product(norm_urm, self.model, k=top_k)
        self.eurm = sps.csr_matrix(self.eurm)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

        return self.eurm

if __name__ == '__main__':
    dr = Datareader(verbose=False, mode='offline', only_load=True)
    urm = dr.get_urm(binary=False)
    pid = dr.get_test_pids()
    position_urm = dr.get_position_matrix(position_type='last')
    pos_urm = position_urm.T.tocoo().tocsr()
    ev = Evaluator(dr)

    knn = 100
    topk = 750

    rec = TF_collaborative_item()
    # for knn in range(50, 300, 50):
    rec.fit(urm, pid)
    rec.compute_model(verbose=True, knn=knn, power=0.6, save_model=False)
    # rec.model = rec.model.tocsr()
    # rec.model.eliminate_zeros()
    #
    # rec.model = position_similarity(rec.model, pos_urm, knn=knn, verbose=True, num_threads=64)

    rec.compute_rating(top_k=topk, verbose=True, small=True)
    ev.evaluate(recommendation_list=eurm_to_recommendation_list(rec.eurm, datareader=dr, remove_seed=True),
                name="TFIDF_item_"+str(knn), old_mode=False)