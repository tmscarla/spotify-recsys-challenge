from utils.definitions import ROOT_DIR
import numpy as np
from tqdm import tqdm
from scipy import sparse
from utils.datareader import Datareader


class HoleBoost:

    def __init__(self, similarity_matrix, eurm, datareader):
        self.similarity_matrix = similarity_matrix
        self.eurm = eurm

        self.test_interactions_df = datareader.get_df_test_interactions()
        self.test_interactions_df.sort_values(['pos'], ascending=True)

        test_playlists_df =  datareader.get_df_test_playlists()
        self.test_playlists = test_playlists_df['pid'].as_matrix()

        # A list of list [pos, tid] for each playlist sorted by pos
        self.known_tracks = self.test_interactions_df.groupby(['pid'])[['pos', 'tid']]\
            .apply(lambda x: x.values.tolist())
        for s in self.known_tracks:
            s = s.sort(key=lambda x: x[0])

    def compute_boost(self, playlist, track):
        """
        Compute the boost for a target playlist and a target track, iterating over all the holes.
        :param playlist: a target playlist
        :param track: a recommended track for playlist
        :return: boost: the estimated boost for the track prediction
        """
        # Initialization
        known_tracks = self.known_tracks[playlist]
        num_holes = len(known_tracks) - 1
        boost = 0
        track_left = known_tracks[0]
        similarity_left = self.similarity_matrix[track, track_left[1]]

        for i in range(0, num_holes):

            track_right = known_tracks[i+1]
            similarity_right = self.similarity_matrix[track, track_right[1]]

            # Compute and increment boost
            if similarity_left > 0:
                similarity = similarity_left * similarity_right
                distance = track_right[0] - track_left[0]
                boost += similarity/distance

            # Update values
            similarity_left = similarity_right
            track_left = track_right

        return boost

    def boost_eurm(self, playlists_indices_to_boost, k, gamma=1):
        """
        Boost the eurm for the playlists in playlists_indices_to_boost.
        :param: playlist_indices_to_boost: the indices wrt the eurm
        :param: k: the first k predicted tracks will be considered for boosting
        :param: gamma: the weight of the boost
        :return: eurm: the boosted eurm
        """
        data = []
        rows = []
        cols = []

        for i in tqdm(playlists_indices_to_boost, desc='HoleBoost'):
            recommended_data = np.squeeze(self.eurm[i].toarray())
            recommended_tracks = np.argsort(recommended_data)[::-1][:k]

            for t in recommended_tracks:
                p = self.test_playlists[i]
                boost = self.compute_boost(p, t)

                data.append(boost)
                rows.append(i)
                cols.append(t)

        eurm_boosted = sparse.csr_matrix((data, (rows, cols)), shape=self.eurm.shape)

        return self.eurm + (eurm_boosted * gamma)
