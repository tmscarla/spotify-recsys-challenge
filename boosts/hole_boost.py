from utils.definitions import ROOT_DIR
import numpy as np
from tqdm import tqdm
from scipy import sparse
import utils.pre_processing as pre
from utils.datareader import Datareader
import time

"""
How to use it:

hb = HoleBoost(similarity, eurm, datareader, norm=norm_l2_row)
eurm_boosted = hb.boost_eurm([8, 10], k, gamma)
"""

class HoleBoost:

    def __init__(self, similarity, eurm, datareader, norm=None):
        """
        :param similarity: a similarity matrix between tracks
        :param eurm: the eurm to boost
        :param norm: normalization function for both eurm and similarity_matrix
        :param datareader: a Datareader object
        """
        self.similarity_matrix = similarity
        self.eurm = eurm
        self.datareader = datareader

        # Normalize eurm and similarity
        if norm is not None:
            self.similarity_matrix = norm(similarity)
            self.eurm = norm(eurm)

        self.test_interactions_df = self.datareader.get_df_test_interactions()
        self.test_interactions_df.sort_values(['pos'], ascending=True)

        test_playlists_df = self.datareader.get_df_test_playlists()
        self.test_playlists = test_playlists_df['pid'].as_matrix()

        # A list of list [pos, tid] for each playlist sorted by pos
        self.known_tracks = self.test_interactions_df.groupby(['pid'])[['pos', 'tid']]\
            .apply(lambda x: x.values.tolist())
        for s in self.known_tracks:
            s = s.sort(key=lambda x: x[0])

    def compute_boost(self, position, offset, similarity_values, distances):
        """
        Compute the boost for a target playlist and a target track, iterating over all the holes.
        :param playlist: a target playlist
        :param track: a recommended track for playlist
        :return: boost: the estimated boost for the track prediction
        """
        boost = 0
        num_holes = offset - 1

        for i in range(0, num_holes):

            similarity = similarity_values[i + (position * offset)] * \
                         similarity_values[i+1 + (position * offset)]

            distance = distances[i + (position * num_holes)]
            if distance > 0:
                # boost += similarity/distance
                boost += similarity

        return boost

    def boost_eurm(self, playlists_indices_to_boost=[], categories=[8, 10], k=200, gamma=10):
        """
        Boost the eurm for the playlists in playlists_indices_to_boost.
        :param: playlist_indices_to_boost: the indices wrt the eurm (10K, 2M)
        :param: list of categories to boost. If set, playlist_indices_to_boost is ignored.
                Category 1, if present, will be excluded.
        :param: k: the first k predicted tracks will be considered for boosting
        :param: gamma: the weight of the boost
        :return: eurm: the boosted eurm
        """
        data = []
        rows = []
        cols = []

        # If categories is set
        if len(categories) > 0:
            # Always remove first category
            if 1 in categories:
                categories.remove(1)

            playlists_indices_to_boost = []

            for cat in categories:
                playlists_indices_to_boost = playlists_indices_to_boost + self.datareader.get_test_pids_indices(cat)

        # Select playlist
        for i in tqdm(playlists_indices_to_boost, desc='HoleBoost'):
            recommended_data = np.squeeze(self.eurm[i].toarray())
            recommended_tracks = np.argsort(recommended_data)[::-1][:k]

            track_indices_all = []
            holes_indices_all = []
            distances_all = []

            for t in recommended_tracks:
                # Initialization
                known_tracks = np.array(self.known_tracks[self.test_playlists[i]])
                num_holes = len(known_tracks) - 1

                track_indices = np.empty(num_holes + 1)
                track_indices.fill(t)

                holes_indices = known_tracks[:, 1]
                distances = np.diff(known_tracks[:, 0])

                track_indices_all += list(track_indices)
                holes_indices_all += list(holes_indices)
                distances_all += list(distances)

            t_array = np.array(track_indices_all)
            h_array = np.array(holes_indices_all)
            t_array = t_array.astype(np.int)
            h_array = h_array.astype(np.int)

            similarity_values = np.ravel(self.similarity_matrix[t_array, h_array])

            # Compute offset
            known_tracks = np.array(self.known_tracks[self.test_playlists[i]])
            offset = len(known_tracks)

            # Select track
            for pos in range(len(recommended_tracks)):
                track = recommended_tracks[pos]
                boost = self.compute_boost(pos, offset, similarity_values, distances_all)

                data.append(boost)
                rows.append(i)
                cols.append(track)

        # Create sparse matrix with boosted values
        eurm_boosted = sparse.csr_matrix((data, (rows, cols)), shape=self.eurm.shape)

        return self.eurm + (eurm_boosted * gamma)
