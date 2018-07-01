from scipy import sparse
from tqdm import tqdm

from utils.datareader import Datareader
from utils.definitions import *
from utils.evaluator import Evaluator
from utils.post_processing import *
from utils.pre_processing import *


class HeteroBoost(object):

    def __init__(self, datareader, eurm, norm=norm_l1_row):

        self.datareader = datareader
        self.eurm = norm(eurm)
        self.urm = self.datareader.get_urm()

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
        self.known_tracks = self.test_interactions_df.groupby(['pid'])['tid'].apply(list)

        self.track_to_artist = datareader.get_track_to_artist_dict()

        tracks_df = datareader.get_df_tracks()
        self.artists = tracks_df['arid'].values
        self.tracks = tracks_df['tid'].values

    def boost_eurm(self, categories=[9], gamma=0.2):
        """
        Boost the eurm for the playlists in specified categories.
        :param categories: the list of categories to boost
        :param: gamma: the weight of the boost
        :return: eurm: the boosted eurm
        """

        self.pids = []
        for cat in categories:
           self.pids = self.pids + list(self.datareader.get_test_pids(cat=cat))

        data = []
        rows = []
        cols = []

        for idx in tqdm(range(len(self.pids)), desc='HeteroBoost'):
            pid = self.pids[idx]

            p_tracks = self.known_tracks[pid]
            p_artists = [self.track_to_artist[t] for t in p_tracks]

            unique_artists = list(set(p_artists))
            if len(unique_artists) <= 2:
                for a in unique_artists:
                    mask = np.isin(self.artists, a)

                    tracks_of_a = self.tracks[mask]

                    rows.extend([self.test_playlists_eurm_idx[pid] for x in range(len(tracks_of_a))])
                    cols.extend(tracks_of_a)
                    data.extend([1 for x in range(len(tracks_of_a))])

        eurm_boosted = sparse.csr_matrix((data, (rows, cols)), shape=self.eurm.shape)

        return self.eurm + (eurm_boosted * gamma)


if __name__ == '__main__':

    dr = Datareader(mode='offline', only_load=True, verbose=False)
    ev = Evaluator(dr)

    # LOAD
    eurm = sparse.load_npz(ROOT_DIR + '/data/ensemble-offline-data-06_24_18-23_53_09.npz')

    eb = HeteroBoost(dr, eurm)
    eurm = eb.boost_eurm(categories=[3, 4, 5, 6, 7, 8, 9, 10], gamma=0.005)

    rec_list = eurm_to_recommendation_list(eurm, datareader=dr)
    ev.evaluate(rec_list, name='ethero')
