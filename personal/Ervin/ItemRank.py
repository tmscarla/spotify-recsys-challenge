import time
import numpy as np
import scipy.sparse as sps
from recommenders.recommender import Recommender
from recommenders.similarity.s_plus import dot_product
from utils.pre_processing import norm_l1_row
from utils.post_processing import eurm_to_recommendation_list
from utils.datareader import Datareader
from utils.evaluator import Evaluator

class ItemRank(Recommender):
    """
    Gori, M. and Pucci, A. ItemRank: a random-walk based scoring algorithm for recommender engines.
    [online] Dl.acm.org. Available at: https://dl.acm.org/citation.cfm?id=1625720
    """

    def __init__(self):
        super()

    def compute_model(self, top_k=100, alpha=0.85, verbose=False, store_graph=False):
        self.alpha = alpha

        if verbose:
            print('[ Building Correlation Graph ]')
            start_time = time.time()

        pop = 1.0 / (self.urm.sum(axis=0).A1 + 1e-8)

        self.corr_graph = dot_product(self.urm.T, self.urm, verbose=verbose, k=top_k)
        self.corr_graph.eliminate_zeros()
        self.corr_graph = (self.corr_graph.multiply(pop.reshape(1,-1))).tocsr()
        self.corr_graph.data *= alpha

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

        if store_graph:
            if verbose:
                print("[ Storing the correlation graph ]")
            sps.save_npz('corr_graph_top'+str(top_k), self.corr_graph)

    def compute_rating(self, top_k=500, verbose=False, small=False, mode="offline", iter=1):
        """
        :param urm: sparse matrix
        :param model: sparse matrix
        :param top_k: int, element to take for each row after fitting process
        :param verbose: boolean, if true print debug information
        :return: sparse matrix, estimated urm
        """

        if small:
            self.small_urm = self.urm[self.pid]
            self.small_urm = sps.csr_matrix(self.small_urm)
            self.small_urm.eliminate_zeros()

        if verbose:
            print('[ Computing Ratings ]')
            start_time = time.time()

        # Compute first pass of ItemRank
        self.dui = norm_l1_row(self.small_urm)
        self.dui.data *= (1-self.alpha)
        self.eurm = self.small_urm.copy()
        self.eurm.data = np.ones(self.small_urm.data.shape[0]) / self.corr_graph.shape[0]

        self.dui = self.dui.T
        self.eurm = self.eurm.T

        # Subsequent iterations of ItemRank
        for _ in range(iter):
            self.eurm = dot_product(self.corr_graph, self.eurm, verbose=verbose, k=top_k) + self.dui

        self.eurm = self.eurm.T
        self.eurm.eliminate_zeros()

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

        return self.eurm.tocsr()

if __name__ == '__main__':
    dr = Datareader(only_load=True, mode='offline', test_num='1', verbose=False)
    pid = dr.get_test_playlists().transpose()[0]
    urm = dr.get_urm()
    ev = Evaluator(dr)

    urm.data = np.ones(urm.data.shape[0])

    IR = ItemRank()
    IR.fit(urm, pid)

    IR.compute_model(verbose=True, top_k=850)
    IR.compute_rating(top_k=750, verbose=True, small=True, iter=2)

    ev.evaluate(recommendation_list=eurm_to_recommendation_list(IR.eurm, remove_seed=True, datareader=dr),
                name="ItemRank", old_mode=False)