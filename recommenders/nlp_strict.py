from recommenders.similarity.dot_product import dot_product
from recommenders.similarity.s_plus import tversky_similarity
from tqdm import tqdm
from scipy import sparse
from utils.definitions import *
from utils.post_processing import *
import numpy as np
import pandas as pd
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.pre_processing import *


class NLPStrict(object):

    def __init__(self, datareader):
        self.datareader = datareader
        self.title_to_idx = dict()

        train_playlists_df = datareader.get_df_train_playlists()
        test_playlists_df = datareader.get_df_test_playlists()
        concat_df = pd.concat([train_playlists_df, test_playlists_df])
        concat_df = concat_df.fillna('')

        if datareader.offline():
            concat_df = concat_df.sort_values(['pid'], ascending=True)

        self.playlists = concat_df['pid'].as_matrix()
        self.playlist_titles = concat_df['name'].as_matrix()
        self.playlist_titles = [(str(x).lower()).encode('unicode-escape').decode('ASCII') for x in self.playlist_titles]

        counter = 0

        for title in self.playlist_titles:
            if title not in self.title_to_idx.keys() and title != '':
                self.title_to_idx[title] = counter
                counter += 1

    def get_UCM(self):
        """
        Build a UCM (playlists, titles) with lowercase titles of playlists and emojis.
        No tokenization or stemming is applied.
        :return: ucm: the user content matrix
        """
        rows = []
        cols = []
        data = []

        print(max(self.playlists))

        for i in tqdm(range(len(self.playlist_titles)), desc='Building UCM'):
            t = self.playlist_titles[i]
            p = self.playlists[i]

            if t != '':
                rows.append(p)
                cols.append(self.title_to_idx[t])
                data.append(1)

        ucm = sparse.csr_matrix((data, (rows, cols)), shape=(max(self.playlists) + 1,
                                                             len(list(self.title_to_idx.keys()))))

        return ucm


if __name__ == '__main__':

    datareader = Datareader(mode='offline', only_load=True)
    evaluator = Evaluator(datareader)

    nlp_lele = sparse.load_npz(ROOT_DIR + '/data/ensemble/nlp_eurm_offline_bm25.npz')
    nlp_strict = sparse.load_npz(ROOT_DIR + '/data/eurm_nlp_strict.npz')
    top_pop = datareader.get_eurm_top_pop()

    top_pop = norm_l1_row(top_pop)
    nlp_lele = norm_l1_row(nlp_lele)
    nlp_strict = norm_l1_row(nlp_strict)

    nlp_fusion = (nlp_lele * 0.6) + (nlp_strict * 0.4)
    sparse.save_npz(ROOT_DIR + '/data/eurm_nlp_fusion_offline.npz', nlp_fusion)
    evaluator.evaluate(eurm_to_recommendation_list(nlp_fusion, datareader=datareader),
                                                           name='nlp_fusion_no_toppop')

    for a in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for b in [0.10, 0.15]:
            nlp_fusion = (nlp_lele * a) + (nlp_strict * (1.0 - a)) + (b * top_pop)

            evaluator.evaluate(eurm_to_recommendation_list(nlp_fusion, datareader=datareader),
                                                           name='nlp_fusion_bm25_'+str(a)+'_'+str(b), do_plot=False)
