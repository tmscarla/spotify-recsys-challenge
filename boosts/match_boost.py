from scipy import sparse
from tqdm import tqdm

from utils.post_processing import *
from utils.pre_processing import *


class MatchBoost(object):

    def __init__(self, datareader, eurm, norm=norm_l2_row, top_k_art=10000, top_k_alb=10000):

        print('MatchBoost Initialization...')
        self.datareader = datareader
        self.eurm = norm(eurm)
        self.top_k_art = top_k_art
        self.top_k_alb = top_k_alb
        self.urm = self.datareader.get_urm()

        # Create dictionary {pid: title}
        train_playlists_df = self.datareader.get_df_train_playlists()
        test_playlists_df = self.datareader.get_df_test_playlists()
        self.test_playlists = test_playlists_df['pid'].as_matrix()
        concat_df = pd.concat([train_playlists_df, test_playlists_df])

        keys = concat_df['pid'].as_matrix()
        values = concat_df['name'].as_matrix()
        self.pid_to_title = dict(zip(keys, values))

        # Stack together train and test interactions
        train_interactions_df = self.datareader.get_df_train_interactions()
        test_interactions_df = self.datareader.get_df_test_interactions()
        interactions_df = pd.concat([train_interactions_df, test_interactions_df], axis=0, join='outer')
        self.playlists = interactions_df['pid'].as_matrix()
        self.tracks = interactions_df['tid'].as_matrix()

        # Load dictionaries
        self.track_to_artist = self.datareader.get_track_to_artist_dict()
        # self.artist_to_tracks = datareader.get_artist_to_tracks_dict()

        self.track_to_album = self.datareader.get_track_to_album_dict()
        # self.album_to_tracks = self.datareader.get_album_to_tracks_dict()

        if datareader.offline():
            self.artist_to_tracks = load_obj(name='artist_tracks_dict_offline', path=ROOT_DIR + '/boosts/')
            self.album_to_tracks = load_obj(name='album_tracks_dict_offline', path=ROOT_DIR + '/boosts/')
        else:
            self.artist_to_tracks = load_obj(name='artist_tracks_dict_online', path=ROOT_DIR + '/boosts/')
            self.album_to_tracks = load_obj(name='album_tracks_dict_online', path=ROOT_DIR + '/boosts/')

        # Gather pids from train and test
        self.pids = list(self.datareader.get_train_pids()) + list(self.datareader.get_test_pids())

        # Ordered list of artists and albums from interactions
        print('Computing list of artists and albums...')
        self.artists = [self.track_to_artist[t] for t in self.tracks]
        self.albums = [self.track_to_album[t] for t in self.tracks]

        # Build popularity
        self.__build_artists_pop()
        self.__build_albums_pop()

        # Compute tracks popularity
        self.tracks_pop = self.urm.sum(axis=0)
        self.tracks_pop = np.ravel(np.asarray(self.tracks_pop))

        # Dictionary that maps each pid of test playlists to its index in the eurm
        # { pid: eurm_index }
        self.test_playlists_eurm_idx = dict()

        for i in range(len(self.test_playlists)):
            pid = self.test_playlists[i]
            self.test_playlists_eurm_idx[pid] = i

    def __build_artists_pop(self):
        # Build auxiliary UCM (playlist, artists)
        print('Building artists popularity...')
        ucm_artists = sparse.csr_matrix((np.ones(len(self.playlists)), (self.playlists, self.artists)),
                                        shape=(np.max(self.pids) + 1, len(self.artists)))
        ucm_artists = ucm_artists.tocsr()
        ucm_artists = ucm_artists[self.pids]

        # Compute artists popularity from UCM
        self.artists_pop = ucm_artists.sum(axis=0)
        self.artists_pop = np.ravel(np.asarray(self.artists_pop))

        # Keep top artists
        self.top_artists = np.argsort(self.artists_pop)[::-1][:self.top_k_art]

    def __build_albums_pop(self):
        # Build auxiliary UCM (playlist, albums)
        print('Building albums popularity...')
        ucm_albums = sparse.csr_matrix((np.ones(len(self.playlists)), (self.playlists, self.albums)),
                                       shape=(np.max(self.pids) + 1, len(self.albums)))
        ucm_albums = ucm_albums.tocsr()
        ucm_albums = ucm_albums[self.pids]

        # Compute artists popularity from UCM
        self.albums_pop = ucm_albums.sum(axis=0)
        self.albums_pop = np.ravel(np.asarray(self.albums_pop))

        # Keep top albums
        self.top_albums = np.argsort(self.albums_pop)[::-1][:self.top_k_alb]

    def create_match_dict_artists(self, test_pids):
        """
        Creates a dict between pids and arids which match exactly.
        :param test_pids: pids to be considered as keys in the dictionary
        :return: match_dict: { pid: [arid_1, ... , arid_n] }
        """

        # Artists
        artists_names = list(self.datareader.get_df_artists()['artist_name'].as_matrix())
        artists_names = np.array([str(x).lower() for x in artists_names])
        artists_names = artists_names[self.top_artists]

        # Dictionary
        match_dict = dict()

        for pid in tqdm(test_pids, desc='Creating match dict artists'):
            title = self.pid_to_title[pid]

            for index in range(len(self.top_artists)):
                arid = self.top_artists[index]

                if title == artists_names[index] and len(title) > 4 and title not in STOP_TITLES:

                    if pid in match_dict.keys():
                        match_dict[pid].append(np.int(arid))
                    else:
                        match_dict[pid] = [np.int(arid)]

        return match_dict

    def create_match_dict_albums(self, test_pids):
        """
        Creates a dict between pids and arids which match exactly.
        :param test_pids: pids to be considered as keys in the dictionary
        :return: match_dict: { pid: [arid_1, ... , arid_n] }
        """

        # Albums
        albums_names = list(self.datareader.get_df_test_albums()['album_name'].as_matrix())
        albums_names = np.array([str(x).lower() for x in albums_names])
        albums_names = albums_names[self.top_artists]

        # Dictionary
        match_dict = dict()

        for pid in tqdm(test_pids, desc='Creating match dict albums'):
            title = self.pid_to_title[pid]

            for index in range(len(self.top_albums)):
                alid = self.top_albums[index]

                if title == albums_names[index] and len(title) > 4 and title not in STOP_TITLES:

                    if pid in match_dict.keys():
                        match_dict[pid].append(np.int(alid))
                    else:
                        match_dict[pid] = [np.int(alid)]

        return match_dict

    def boost_eurm(self, categories='all', k_art=50, k_alb=50, gamma_art=0.1, gamma_alb=0.1):
        """
        Boost the eurm for the playlists of the specified categories.
        :param categories: 'all' or a list of int between 1 and 10
        :param k_art: boost only the top popular k tracks for matched artist(s)
        :param k_alb: boost only the top popular k tracks for matched album(s)
        :param gamma_art: boost weight for artists matches
        :param gamma_alb: boost weight for albums matches
        :return: (eurm_boosted, pids_boosted):
        """

        # BOOST INIT

        if categories == 'all':
            self.test_pids = self.datareader.get_test_pids()

        else:
            self.test_pids = []
            for cat in categories:
               self.test_pids = self.test_pids + list(self.datareader.get_test_pids(cat=cat))

        # ARTISTS

        # Create dict
        match_dict_artists = self.create_match_dict_artists(self.test_pids)

        # Init data structures to build the boosting eurm
        data = []
        rows = []
        cols = []

        for pid in tqdm(match_dict_artists.keys()):
            artists_pop_percentage = [self.artists_pop[arid] for arid in match_dict_artists[pid]]
            artists_pop_percentage = np.array(artists_pop_percentage).astype(np.int)

            if len(artists_pop_percentage) == 1:
                artists_pop_percentage = np.reshape(artists_pop_percentage, (1, -1))
            else:
                artists_pop_percentage = np.reshape(artists_pop_percentage, (-1, 1))

            artists_pop_percentage = norm_l1_col(artists_pop_percentage)

            for i in range(len(match_dict_artists[pid])):
                arid = match_dict_artists[pid][i]

                arid_tracks = np.array(self.artist_to_tracks[arid])
                arid_tracks_pop = [self.tracks_pop[t] for t in arid_tracks]

                arid_percentage = int(k_art * artists_pop_percentage[i])
                top_indices = np.argsort(arid_tracks_pop)[::-1][:arid_percentage]
                tracks_boost = arid_tracks[top_indices]

                for t in tracks_boost:
                    data.append(1)
                    rows.append(self.test_playlists_eurm_idx[pid])
                    cols.append(t)

        eurm_boosting_artists = sparse.csr_matrix((data, (rows, cols)), shape=self.eurm.shape)

        # ALBUMS

        # Create dict
        match_dict_albums = self.create_match_dict_albums(self.test_pids)

        # Init data structures to build the boosting eurm
        data = []
        rows = []
        cols = []

        for pid in tqdm(match_dict_albums.keys()):
            albums_pop_percentage = [self.albums_pop[alid] for alid in match_dict_albums[pid]]
            albums_pop_percentage = np.array(albums_pop_percentage).astype(np.int)

            if len(albums_pop_percentage) == 1:
                albums_pop_percentage = np.reshape(albums_pop_percentage, (1, -1))
            else:
                albums_pop_percentage = np.reshape(albums_pop_percentage, (-1, 1))

            albums_pop_percentage = norm_l1_col(albums_pop_percentage)

            for i in range(len(match_dict_albums[pid])):
                alid = match_dict_albums[pid][i]

                alid_tracks = np.array(self.album_to_tracks[alid])
                alid_tracks_pop = [self.tracks_pop[t] for t in alid_tracks]

                alid_percentage = int(k_alb * albums_pop_percentage[i])
                top_indices = np.argsort(alid_tracks_pop)[::-1][:alid_percentage]
                tracks_boost = alid_tracks[top_indices]

                for t in tracks_boost:
                    data.append(1)
                    rows.append(self.test_playlists_eurm_idx[pid])
                    cols.append(t)

        eurm_boosting_albums = sparse.csr_matrix((data, (rows, cols)), shape=self.eurm.shape)

        # Return boosted eurm and list of boosted pids
        eurm_boosted = self.eurm + (eurm_boosting_artists * gamma_art) + (eurm_boosting_albums * gamma_alb)
        pids_boosted = list(set(list(match_dict_artists.keys()) + list(match_dict_albums.keys())))

        return eurm_boosted, pids_boosted

