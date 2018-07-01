from scipy import sparse
from tqdm import tqdm

from utils.post_processing import *
from utils.pre_processing import *


class TopBoost(object):

    def __init__(self, datareader, eurm, similarity, norm=norm_l1_row):

        self.datareader = datareader
        self.eurm = norm(eurm)
        self.similarity = similarity
        self.urm = self.datareader.get_urm()

        # Compute popularity
        self.popularity = self.urm.sum(axis=0)
        self.popularity = np.squeeze(np.asarray(self.popularity))

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

        self.track_to_artist = datareader.get_track_to_artist_dict()
        # self.artist_to_tracks = datareader.get_artist_to_tracks_dict()
        self.artist_to_tracks = load_obj(path=ROOT_DIR + '/boosts/', name='artist_tracks_dict_offline')

        self.tracks = datareader.get_df_tracks()['tid'].as_matrix()
        self.artists = np.array([self.track_to_artist[t] for t in self.tracks])


    def boost_eurm(self, categories=[9], top_k=100, gamma=0.01):
        """
        Boost the eurm for the playlists in specified categories.
        :param categories: the list of categories to boost
        :param last_tracks: list of last tracks that will be boosted in each category
        :param: k: the first top k tracks of the artist will be boosted
        :param: gamma: the weight of the boost
        :return: eurm: the boosted eurm
        """

        self.pids = []
        for cat in categories:
           self.pids = self.pids + list(self.datareader.get_test_pids(cat=cat))

        data = []
        rows = []
        cols = []

        for pid in tqdm(self.pids, desc='TopBoost'):

            tracks = [t[1] for t in self.known_tracks[pid]]

            artists = [self.track_to_artist[t] for t in tracks]
            unique_artists = len(set(artists))

            if unique_artists > 70:
                top_tracks = np.argsort(self.popularity)[::-1][:top_k]


            # first_track = self.known_tracks[pid][-1][1]
            # first_artist = self.track_to_artist[first_track]
            # tracks = np.array(self.artist_to_tracks[first_artist])
            #
            # mask = np.where(self.artists == first_artist)
            #
            # similarity_row = self.similarity[first_track, :].toarray().ravel()
            # similar_indices = np.argsort(similarity_row[mask])[::-1][:top_k]
            #
            # similar_tracks = tracks[similar_indices]

                for t in top_tracks:
                    data.append(1)
                    rows.append(self.test_playlists_eurm_idx[pid])
                    cols.append(t)

        eurm_boosted = sparse.csr_matrix((data, (rows, cols)), shape=self.eurm.shape)

        return self.eurm + (eurm_boosted * gamma)



