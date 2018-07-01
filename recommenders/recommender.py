import time
import scipy.sparse as sps
from recommenders.similarity.similarity import *
from utils.post_processing import eurm_remove_seed


class Recommender(object):

    def __init__(self):
        self.urm = None
        self.model = None

    def fit(self, urm, pid):
        self.urm = sps.csr_matrix(urm)
        self.pid = pid

    def _fit(self, urm, model, pid):
        self.urm = sps.csr_matrix(urm)
        self.pid = pid
        self.model = model



    def compute_rating(self, urm2=None, datareader=None, top_k=750, verbose=False, small=False, mode="offline", remove_seed=True):
        """
        :param urm: sparse matrix
        :param model: sparse matrix
        :param top_k: int, element to take for each row after fitting process
        :param small: boolean, if true return an eurm matrix with just the target playlist
        :param verbose: boolean, if true print debug information
        :param remove_seed: boolean, if true remove seed from eurm
        :return: sparse matrix, estimated urm
        """
        if small:
            self.urm = sps.csr_matrix(self.urm[self.pid])
        self.urm = sps.csr_matrix(self.urm)
        self.model = sps.csr_matrix(self.model)

        if verbose:
            print("[ Compute ratings ]")

            start_time = time.time()


        if urm2 != None:
            self.urm = urm2[self.pid]
        self.eurm = dot(self.urm, self.model,  k=top_k)

        print("eurm shape: " + str(self.eurm.shape))

        if remove_seed:
            if datareader is None:
                print('[ WARNING! Datareader is None in "compute rating". mode is set to'+mode.upper()+', creating it again. '
                            'A future version will require it. ]')
                from utils.datareader import Datareader
                datareader = Datareader(mode=mode, only_load=True)
            self.eurm = eurm_remove_seed(self.eurm, datareader=datareader)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

        return self.eurm.tocsr()
