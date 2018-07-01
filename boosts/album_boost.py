from scipy import sparse
from tqdm import tqdm

from utils.datareader import Datareader
from utils.definitions import *
from utils.evaluator import Evaluator
from utils.post_processing import *
from utils.pre_processing import *


class AlbumBoost(object):

    def __init__(self, datareader, eurm, norm=norm_l1_row):

        self.datareader = datareader
        self.eurm = norm(eurm)
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
        self.track_to_album = datareader.get_track_to_album_dict()

        if datareader.online():
            self.album_to_tracks = load_obj(name='album_tracks_dict_online', path=ROOT_DIR + '/boosts/dict/')
        else:
            self.album_to_tracks = load_obj(name='album_tracks_dict_offline', path=ROOT_DIR + '/boosts/dict/')

    def boost_eurm(self, categories=[9], top_k=[40], gamma=1):
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

        for idx in tqdm(range(len(self.pids)), desc='AlbumBoost'):
            pid = self.pids[idx]
            cat_idx = int(idx / 1000)

            lasts = []
            for j in range(2):
                lasts.append(self.track_to_album[self.known_tracks[pid][-j-1][1]])

            lasts = list(set(lasts))

            if len(lasts) == 1:
                tracks = np.array(self.album_to_tracks[lasts[0]])

                pop_indices = np.argsort(self.popularity[tracks])[::-1][:top_k[cat_idx]]
                tracks_pop = tracks[pop_indices]

                for t in tracks_pop:
                    data.append(1)
                    rows.append(self.test_playlists_eurm_idx[pid])
                    cols.append(t)

        eurm_boosted = sparse.csr_matrix((data, (rows, cols)), shape=self.eurm.shape)

        return self.eurm + (eurm_boosted * gamma)
