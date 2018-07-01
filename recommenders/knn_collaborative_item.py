import time
from recommenders.recommender import Recommender
from recommenders.similarity.similarity import *

class Knn_collaborative_item(Recommender):
    def __init__(self):
        super()

    def compute_model(self, top_k=50, sm_type="cosine", shrink=0, alpha=0, beta=0, threshold=0, binary = False, verbose=False, p1=1, p2=1, w1=1, w2=1, l=0.5, c=0.5, t1=1, t2=1):

        """
        :param matrix: sparse matrix, urm for knn item, p3alpha, p3beta, urm.T for knn user
        :param top_k: int, element to take for each row after model computation problem
        :param sm_type: string, similarity to use (use constant in this class to specify)
        :param shrink: float, shrink term for the similarity
        :param alpha: float, parameter used for asimmetric cosine, p3alpha, rp3beta and tversky
        :param beta:  float, parameter used rp3beta and tversky
        :param threshold: float, threshold to cut similarity value after computation
        :param verbose: boolen, if true print debug information
        :param binary: boolen, if true al the data in the matrix are transformed to binary data (0 or 1)
        :return: sparse matrix, model for all the similarity
        """

        #TODO: remove after update
        import warnings
        warnings.warn('This function still use the old version of the metrics, not the s_plus ones')

        if verbose:
            print("[ Creating model with " + sm_type + " similarity ]")
            start_time = time.time()

        if sm_type == COSINE:
            self.model = cosine(self.urm.T, k=top_k, shrink=shrink, threshold=threshold, binary=binary)

        elif sm_type == JACCARD:
            self.model = jaccard(self.urm.T, k=top_k, shrink=shrink, threshold=threshold, binary=binary)

        elif sm_type == TANIMOTO:
            self.model = tanimoto(self.urm.T, k=top_k, shrink=shrink, threshold=threshold, binary=binary)

        elif sm_type == AS_COSINE:
            self.model = cosine(self.urm.T, alpha=alpha, k=top_k, shrink=shrink)

        elif sm_type == DICE:
            self.model = dice(self.urm.T, k=top_k, shrink=shrink, threshold=threshold, binary=binary)

        elif sm_type == TVERSKY:
            self.model = tversky(self.urm.T, k=top_k, alpha=alpha, beta=beta, shrink=shrink, binary=binary)

        elif sm_type == SPLUS:
            self.model = s_plus(self.urm.T, self.urm,
                                weight_pop_m1='sum', weight_pop_m2='sum',
                                weight_feature_m1='sum', weight_feature_m2='sum',
                                p1=p1, p2=p2,
                                w1=w1, w2=w2,
                                normalization=False,
                                l=l,
                                c=c,
                                t1=t1, t2=t2,
                                k=top_k, shrink=shrink, threshold=threshold,
                                binary=binary,
                                target_items=None
                                )

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

