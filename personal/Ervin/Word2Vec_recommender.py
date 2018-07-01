import time
import numpy as np
import scipy.sparse as sps
from gensim.models import Word2Vec
from tqdm import tqdm
from recommenders.recommender import Recommender
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.post_processing import eurm_to_recommendation_list
from recommenders.similarity.s_plus import dot_product

class W2VRecommender(Recommender):
    """
    Requires gensim package: pip install gensim
    """

    RECOMMENDER_NAME = "W2VRecommender"

    def __init__(self):
        super()

    def compute_model(self, negative=5, sg=1, size=50, min_count=1, workers=64, iter=1, window=None, verbose=False):
        sentences = []
        for row in tqdm(range(self.urm.shape[0]), desc='Generating sentences'):
            words = self.urm.indices[self.urm.indptr[row]:self.urm.indptr[row+1]]
            words = words.astype(np.str)
            if len(words) > 0:
                sentences.append(words.tolist())

        if verbose:
            print('[ Building Word2Vec model ]')
            start_time = time.time()

        if window is None:
            window = np.max(self.urm.sum(axis=1).A1)

        w2v = Word2Vec(sentences=sentences, sg=sg, size=size, min_count=min_count, workers=workers, iter=iter,
                       seed=123, negative=negative, window=window)
        w2v.init_sims(replace=True)
        self.kv = w2v.wv

        # if verbose:
        #     print('[ Building Similarity Matrix ]')
        #
        # syn0norm = sps.csr_matrix(self.kv.syn0norm)
        # self.model = dot_product(syn0norm, syn0norm.T, k=850)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

    def compute_rating(self, verbose=False, small=False, mode="offline", top_k=750):
        if small:
            self.urm = sps.csr_matrix(self.urm)[self.pid]
            self.eurm = sps.lil_matrix(self.urm.shape, dtype=np.float32)

        if verbose:
            print('[ Computing ratings ]')
            start_time = time.time()

        for row in tqdm(range(1000, self.urm.shape[0]), desc='Calculating similarities'):
            test_words = self.urm.indices[self.urm.indptr[row]:self.urm.indptr[row+1]]
            test_words = test_words.astype(np.str)
            most_sim = self.kv.most_similar(positive=test_words, topn=top_k)
            tracks = [tup[0] for tup in most_sim]
            sim = [tup[1] for tup in most_sim]
            self.eurm[row, tracks] = sim

        self.eurm = self.eurm.tocsr()
        self.eurm.eliminate_zeros()

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

    # def compute_rating2(self, verbose=False, small=False, mode="offline", remove_seed=True):
    #     if small:
    #         self.urm = sps.csr_matrix(self.urm)[self.pid]
    #         self.eurm = sps.lil_matrix(self.urm.shape, dtype=np.float32)
    #
    #     if verbose:
    #         print('[ Computing ratings ]')
    #         start_time = time.time()
    #
    #     for row in tqdm(range(1000, self.urm.shape[0]), desc='Calculating similarities'):
    #         test_words = self.urm.indices[self.urm.indptr[row]:self.urm.indptr[row+1]]
    #         test_words = test_words.astype(np.str)
    #         for w in test_words:
    #             most_sim = self.kv.most_similar(positive=w, topn=500)
    #             tracks = [tup[0] for tup in most_sim]
    #             sim = [tup[1] for tup in most_sim]
    #             self.eurm[row, tracks] = self.eurm[row, tracks].toarray() + sim
    #
    #     print(self.eurm.shape)
    #     self.eurm = self.eurm.tocsr()
    #     self.eurm.eliminate_zeros()
    #
    #     if verbose:
    #         print("time: " + str(int(time.time() - start_time) / 60))


if __name__ == '__main__':
    dr = Datareader(only_load=True, mode='offline', test_num='1', verbose=False)
    pid = dr.get_test_playlists().transpose()[0]
    urm = dr.get_urm()

    urm.data = np.ones(urm.data.shape[0])

    ev = Evaluator(datareader=dr)

    model = W2VRecommender()
    model.fit(urm, pid)
    model.compute_model(verbose=True, size=50)
    model.compute_rating(verbose=True, small=True, top_k=750)
    ev.evaluate(recommendation_list=eurm_to_recommendation_list(model.eurm, remove_seed=True, datareader=dr),
                name="W2V", old_mode=False)