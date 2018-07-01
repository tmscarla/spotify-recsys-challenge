import time
from recommenders.recommender import Recommender
from recommenders.similarity.similarity import *


class R_p_3_beta(Recommender):
    def __init__(self):
        self.p_ui = None
        self.p_iu = None
        self.model = None
        self.eurm = None
        self.pop = None

    def fit(self, p_ui, p_iu, pop, pid):
        self. urm = p_ui
        self.pid = pid
        self.p_ui = p_ui
        self.p_iu = p_iu
        self.pop = pop

    def compute_model(self, top_k=50, shrink=0, alpha=0, beta=0, threshold=0, verbose=False):
        """
        :param matrix: sparse matrix, urm for knn item, p3alpha, p3beta, urm.T for knn user
        :param top_k: int, element to take for each row after model computation problem
        :param sm_type: string, similarity to use (use constant in this class to specify)
        :param shrink: float, shrink term for the similarity
        :param alpha: float, parameter used for asimmetric cosine, p3alpha, rp3beta and tversky
        :param beta:  float, parameter used rp3beta and tversky
        :param threshold: float, threshold to cut similarity value after computation
        :param verbose: boolena, if true print debug information
        :return: sparse matrix, model for all the similarity
        """

        #TODO: remove after update
        import warnings
        warnings.warn('This function still use the old version of the metrics, not the s_plus ones')

        if verbose:
            print("[ Creating model with rp3beta similarity ]")
            start_time = time.time()

        self.model = rp3beta(self.p_iu, self.p_ui, self.pop, k=top_k, alpha=alpha, beta=beta, shrink=shrink, threshold=threshold)


        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

