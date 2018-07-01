"""
@author Ervin Dervishaj
@email vindervishaj@gmail.com
"""

import time
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from recommenders.similarity.s_plus import dot_product
from recommenders.recommender import Recommender
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.post_processing import eurm_to_recommendation_list
from utils.pretty_printer import Pretty_printer

class Doc2Vec_recommender(Recommender):
    def __init__(self, datareader):
        self.pr = Pretty_printer(datareader)
        self.playlist_id_to_name_dict = self.pr.playlist_id_to_name_dict

    def compute_model(self, topn=100, dm=1, size=50, negative=5, window=None, min_count=1, iter=1, workers=64,
                      verbose=False):
        max_window = -1
        sentences = []
        for key in tqdm(self.playlist_id_to_name_dict, desc='Creating sentences'):
            label = 'PLS_' + str(key)
            s = str(self.playlist_id_to_name_dict[key]).split(' ')
            # words = self.urm.indices[self.urm.indptr[key]:self.urm.indptr[key + 1]]
            # words = words.astype(np.str).tolist()
            # s.extend(words)
            sentences.append(TaggedDocument(s, [label]))
            max_window = max([max_window, len(s)])

        if window == None:
            window = max_window

        if verbose:
            print('[ Building Doc2Vec model ]')
            start_time = time.time()

        self.d2v = Doc2Vec(documents=sentences, dm=dm, size=size, min_count=min_count, epochs=iter, window=window,
                             workers=workers, negative=negative)
        # self.model.init_sims(replace=True)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

        self.most_sim(topn=topn, verbose=verbose)

    def most_sim(self, verbose=False, topn=100, mode='offline'):
        self.model = sps.lil_matrix((self.urm.shape[0], self.urm.shape[0]), dtype=np.float32)

        playlist_test_df = self.pr.dr.get_df_test_playlists()
        pl_pid = list(playlist_test_df['pid'].as_matrix())
        pl_names = list(playlist_test_df['name'].as_matrix())
        test_pls_id_to_name_dict = dict(zip(pl_pid, pl_names))

        if verbose:
            print('[ Computing similarity ]')
            start_time = time.time()

        for key in tqdm(test_pls_id_to_name_dict, desc='Computing Ratings'):
            s = str(test_pls_id_to_name_dict[key]).split(' ')
            # words = self.urm.indices[self.urm.indptr[key]:self.urm.indptr[key + 1]]
            # words = words.astype(np.str).tolist()
            # s.extend(words)
            s_vec = self.d2v.infer_vector(s)
            most_sim = self.d2v.docvecs.most_similar(positive=[s_vec], topn=topn)
            pls = [int(tup[0][4:]) for tup in most_sim]
            sim = [tup[1] for tup in most_sim]
            self.model[key, pls] = sim

        self.model = self.model.tocsr()

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

    def compute_rating(self, top_k=750, verbose=False, small=False):
        if small:
            self.model = self.model[self.pid]

        if verbose:
            print('[ Computing ratings ]')
            start_time = time.time()

        self.eurm = dot_product(self.model, self.urm, verbose=True, k=top_k)

        if verbose:
            print("time: " + str(int(time.time() - start_time) / 60))

        return self.eurm


if __name__ == '__main__':
    dr = Datareader(verbose=True, only_load=True, mode='offline', test_num='1')
    ev = Evaluator(datareader=dr)
    pid = dr.get_test_pids()
    urm = dr.get_urm(binary=True)

    rec = Doc2Vec_recommender(datareader=dr)
    rec.fit(urm, pid)
    rec.compute_model(verbose=True, topn=100)
    rec.compute_rating(verbose=True, small=True)
    ev.evaluate(recommendation_list=eurm_to_recommendation_list(rec.eurm, remove_seed=True, datareader=dr),
                name="D2V", old_mode=False)