from scipy import sparse
from tqdm import tqdm

from utils.pre_processing import *


class TailBoost(object):

    def __init__(self, datareader, eurm, similarity, norm=norm_l2_row):

        self.datareader = datareader
        self.eurm = norm(eurm)
        self.similarity = norm(similarity)

        self.test_interactions_df = self.datareader.get_df_test_interactions()
        self.test_interactions_df.sort_values(['pos'], ascending=True)

        test_playlists_df = self.datareader.get_df_test_playlists()
        self.test_playlists = test_playlists_df['pid'].as_matrix()

        # Dictionary that maps each pid of test playlists to its index in the eurm
        # { pid: eurm_index }
        self.test_playlists_eurm_idx = dict()

        for i in range(len(self.test_playlists)):
            pid = self.test_playlists[i]
            self.test_playlists_eurm_idx[pid] = i

        # A list of list [pos, tid] for each playlist sorted by pos
        self.known_tracks = self.test_interactions_df.groupby(['pid'])[['pos', 'tid']]\
            .apply(lambda x: x.values.tolist())
        for s in self.known_tracks:
            s = s.sort(key=lambda x: x[0])

    def boost_eurm(self, categories, last_tracks, k, gamma):
        """
        Boost the eurm for the playlists in specified categories.
        :param categories: the list of categories to boost
        :param last_tracks: list of last tracks that will be boosted in each category
        :param: k: the first k simile tracks will be considered for boosting
        :param: gamma: the weight of the boost
        :return: eurm: the boosted eurm
        """

        self.pids = []
        for c in categories:
            self.pids = self.pids + list(self.datareader.get_test_pids(cat=c))

        data = []
        rows = []
        cols = []

        for idx in tqdm(range(len(self.pids)), desc='TailBoost'):
            pid = self.pids[idx]
            cat_index = int(idx / 1000)

            # Compute known tracks and invert them from last to first
            known_tracks = self.known_tracks[pid][::-1][:last_tracks[cat_index]]

            for track in known_tracks:
                # Slice row
                row_start = self.similarity.indptr[track[1]]
                row_end = self.similarity.indptr[track[1] + 1]

                row_columns = self.similarity.indices[row_start:row_end]
                row_data = self.similarity.data[row_start:row_end]

                # Compute top k simile tracks for track
                top_k = np.argsort(row_data, kind='mergesort')[::-1][:k[cat_index]]
                indices_to_boost = row_columns[top_k]
                boost_values = row_data[top_k]

                for i in range(len(indices_to_boost)):
                    index = indices_to_boost[i]

                    weighted_boost_value = boost_values[i] * log(track[0] + 1)

                    data.append(weighted_boost_value * gamma[cat_index])
                    rows.append(self.test_playlists_eurm_idx[pid])
                    cols.append(index)

        eurm_boosted = sparse.csr_matrix((data, (rows, cols)), shape=self.eurm.shape)

        return self.eurm + eurm_boosted
