import time
from recommenders.recommender import Recommender
from recommenders.similarity.similarity import *


class knn_content_user(Recommender):
    def __init__(self):
        super()
        self.ucm = None

    def fit(self, urm, ucm):
        super(urm)
        self.ucm = ucm

    def compute_model(self, top_k=50, sm_type="cosine", shrink=0, alpha=0, beta=0, threshold=0, verbose=False):
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
            print("[ Creating model with " + sm_type + " similarity ]")
            start_time = time.time()

        if sm_type == COSINE:
            self.model = cosine(self.ucm, k=top_k, shrink=shrink, threshold=threshold, binary=False)

        elif sm_type == JACCARD:
            self.model = jaccard(self.ucm, k=top_k, shrink=shrink, threshold=threshold, binary=False)

        elif sm_type == TANIMOTO:
            self.model = tanimoto(self.ucm, k=top_k, shrink=shrink, threshold=threshold, binary=False)

        elif sm_type == AS_COSINE:
            self.model = cosine(self.ucm, alpha=alpha, k=top_k, shrink=shrink)

        elif sm_type == DICE:
            self.model = dice(self.ucm, k=top_k, shrink=shrink, threshold=threshold, binary=False)

        elif sm_type == TVERSKY:
            self.model = tversky(self.ucm, alpha=alpha, beta=beta, shrink=shrink)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

    def compute_rating(self, top_k=500, verbose=False, small=False):
        """
        :param urm: sparse matrix
        :param model: sparse matrix
        :param top_k: int, element to take for each row after fitting process
        :param verbose: boolean, if true print debug information
        :return: sparse matrix, estimated urm
        """

        if small:
            self.urm = sp.csr_matrix(self.urm)[self.pid]
        self.urm = self.urm.to_csr()
        self.model = self.model.to_csr()

        if verbose:
            print("[ Compute ratings ]")
            start_time = time.time()

        self.eurm = dot(self.model.T, self.urm, k=top_k)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))



