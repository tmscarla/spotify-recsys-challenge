import pandas as pd
import numpy as np
import nltk
from nltk import stem
from nltk.tokenize import RegexpTokenizer
from utils.datareader import Datareader
from recommenders.similarity.dot_product import dot_product
from recommenders.similarity.tversky import tversky_similarity
from tqdm import tqdm
from scipy import sparse
from utils.definitions import *


class NLP(object):

    def __init__(self, mode, datareader, stopwords, verbose=True):
        """
        :param mode: ['both', 'tracks', 'playlists']
                     Consider tokens extracted just from tracks, playlists or both.
                     Of course, in order to compute the UCM, playlists titles are needed,
                     as well as tracks titles are needed in case of the ICM.
        :param datareader: a Datareader object
        :param stopwords: a list of stopwords
        :param verbose: set verbosity
        """

        self.mode = mode
        self.datareader = datareader

        # Read dataframes and put together test and train
        tracks_df = self.datareader.get_df_tracks()
        train_playlists_df = self.datareader.get_df_train_playlists()
        test_playlists_df = self.datareader.get_df_test_playlists()
        concat_df = pd.concat([train_playlists_df, test_playlists_df])

        if datareader.offline():
            concat_df = concat_df.sort_values(['pid'], ascending=True)

        # Dictionary that maps each pid of test playlists to its index in the eurm
        # { pid: eurm_index }
        self.test_playlists_eurm_idx = dict()
        self.test_playlists = test_playlists_df['pid'].as_matrix()

        for i in range(len(self.test_playlists)):
            pid = self.test_playlists[i]
            self.test_playlists_eurm_idx[pid] = i

        self.stopwords = stopwords
        self.playlists = concat_df['pid'].as_matrix()
        self.playlist_titles = concat_df['name'].as_matrix()
        self.tracks_titles = tracks_df['track_name'].as_matrix()
        self.both_titles = np.concatenate((self.playlist_titles, self.tracks_titles))

        # Known tracks
        test_interactions_df = datareader.get_df_test_interactions()
        self.known_tracks = test_interactions_df.groupby(['pid'])['tid'].apply(list)
        self.playlists_with_tracks = set(self.known_tracks.index.values)

        # Genres dictionaries
        keys = GENRES
        values = [set() for i in range(len(GENRES))]
        self.genres = dict(zip(keys, values))
        self.genres_playlists = dict(zip(keys, values))

        # {token: set(p1, p2, ...) }
        self.tokens_playlist_dict = dict()
        # {token: set(t1, t2, ...) }
        self.tokens_track_dict = dict()
        # {token: [set(p1, p2, ...), set(t1, t2, ...)] }
        self.tokens_both_dict = dict()

        # Set mode
        if self.mode == 'both':
            self.__set_params_both()
        elif self.mode == 'tracks':
            self.__set_params_tracks()
        elif self.mode == 'playlists':
            self.__set_params_playlist()
        else:
            try:
                raise ValueError('Error: mode must be one of = ["both", "tracks", "playlists"]')
            except ValueError as error:
                print(repr(error))

        # Save dictionary keys in a list
        self.tokens_playlist = list(self.tokens_playlist_dict.keys())
        self.tokens_track = list(self.tokens_track_dict.keys())
        self.tokens_both = list(self.tokens_both_dict.keys())

        self.verbose = verbose

    def __set_params_playlist(self):

        ps = stem.PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')

        for i in tqdm(range(len(self.playlist_titles)), desc='Playlist titles extraction'):
            title = self.playlist_titles[i]

            if type(title) is str:
                tokens = tokenizer.tokenize(title)

                for token in tokens:
                    token.lower()
                    token = token.replace("_", "")
                    if token not in self.stopwords and len(token) > 1:
                        s = ps.stem(token)

                        if s in self.tokens_playlist_dict.keys():
                            self.tokens_playlist_dict[s].add(i)
                        else:
                            self.tokens_playlist_dict[s] = {i}

                        # Genres matching
                        p = self.playlists[i]
                        if token in GENRES and p in self.playlists_with_tracks:
                            self.genres_playlists[token].add(p)

                            for track in self.known_tracks[p]:
                                self.genres[token].add(track)


    def __set_params_tracks(self):

        ps = stem.PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')

        for i in tqdm(range(len(self.tracks_titles)), desc='Tracks titles extraction'):
            title = self.tracks_titles[i]

            if type(title) is str:
                tokens = tokenizer.tokenize(title)

                for token in tokens:
                    token.lower()
                    token = token.replace("_", "")
                    if token not in self.stopwords and len(token) > 1:
                        s = ps.stem(token)

                        if s in self.tokens_track_dict.keys():
                            self.tokens_track_dict[s].add(i)
                        else:
                            self.tokens_track_dict[s] = {i}

    def __set_params_both(self):

        ps = stem.PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')

        for i in tqdm(range(len(self.both_titles)),
                      desc='Playlist and tracks titles extraction'):
            title = self.both_titles[i]

            if type(title) is str:
                tokens = tokenizer.tokenize(title)

                for token in tokens:
                    token.lower()
                    token = token.replace("_", "")
                    if token not in self.stopwords and len(token) > 1:
                        s = ps.stem(token)

                        # Currently iterating playlists titles
                        if i < len(self.playlist_titles):

                            # Token already in dictionary
                            if s in self.tokens_both_dict.keys():
                                self.tokens_both_dict[s][0].add(i)
                            else:
                                self.tokens_both_dict[s] = [{i}, set()]

                        # Currently iterating tracks titles
                        else:

                            # Token already in dictionary
                            if s in self.tokens_both_dict.keys():
                                self.tokens_both_dict[s][1].add(i - len(self.playlist_titles))
                            else:
                                self.tokens_both_dict[s] = [set(), {i - len(self.playlist_titles)}]

    def get_ucm(self):
        rows = []
        cols = []
        data = []

        # Use only playlists titles
        if self.mode == 'playlists':
            for i in tqdm(range(len(self.tokens_playlist)), desc='Build UCM playlists titles'):
                word = self.tokens_playlist[i]

                for p in self.tokens_playlist_dict[word]:
                    rows.append(p)
                    cols.append(i)
                    data.append(1)

            self.ucm = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.playlist_titles),
                                                                      len(self.tokens_playlist)))
        # Use playlists + tracks titles
        elif self.mode == 'both':
            for i in tqdm(range(len(self.tokens_both)), desc='Build UCM all titles'):
                word = self.tokens_both[i]

                for p in self.tokens_both_dict[word][0]:
                    rows.append(p)
                    cols.append(i)
                    data.append(1)

            self.ucm = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.playlist_titles),
                                                                      len(self.tokens_both)))

        return self.ucm

    def get_similarity_from_ucm(self):
        self.get_ucm()

        if self.verbose:
            print('Computing similarity from ucm...')

        self.similarity_ucm = tversky_similarity(self.ucm, shrink=200, alpha=0.1, beta=1)
        self.similarity_ucm = self.similarity_ucm.tocsr()

        return self.similarity_ucm

    def get_eurm_from_ucm(self, urm, test_playlists):
        """
        Compute the UCM, then the similarity and return the EURM sliced for test playlists.
        :param test_playlists: the pids of the test playlists
        :return: eurm: the estimated eurm of shape (10K, 2M)
        """
        self.urm = urm
        self.get_similarity_from_ucm()

        if self.verbose:
            print('Computing eurm from ucm...')

        self.eurm = dot_product(self.similarity_ucm, self.urm, k=500)
        self.eurm = self.eurm.tocsr()

        if self.datareader.__online():
            self.eurm = self.eurm[-10000:, :]
        else:
            self.eurm = self.eurm[test_playlists, :]

        return self.eurm

    def get_icm(self):
        rows = []
        cols = []
        data = []

        # Use only tracks titles
        if self.mode == 'tracks':
            for i in tqdm(range(len(self.tokens_track)), desc='Build ICM tracks titles'):
                word = self.tokens_track[i]

                for t in self.tokens_track_dict[word]:
                    rows.append(t)
                    cols.append(i)
                    data.append(1)

            self.icm = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.tracks_titles),
                                                                      len(self.tokens_track)))
        # Use playlists + tracks titles
        elif self.mode == 'both':
            for i in tqdm(range(len(self.tokens_both)), desc='Build ICM all titles'):
                word = self.tokens_both[i]

                for t in self.tokens_both_dict[word][1]:
                    rows.append(t)
                    cols.append(i)
                    data.append(1)

            self.icm = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.tracks_titles),
                                                                      len(self.tokens_both)))
        return self.icm

    def get_similarity_from_icm(self):
        self.get_icm()

        if self.verbose:
            print('Computing similarity from icm...')

        self.similarity_icm = tversky_similarity(self.icm, shrink=200, alpha=0.1, beta=1)
        self.similarity_icm = self.similarity_icm.tocsr()

        return self.similarity_icm

    def get_eurm_from_icm(self, urm, test_playlists):
        """
        Compute the ICM, then the similarity and return the EURM sliced for test playlists.
        :param test_playlists: the pids of the test playlists
        :return: eurm: the estimated eurm of shape (10K, 2M)
        """
        self.urm = urm
        self.get_similarity_from_icm()

        if self.verbose:
            print('Computing similarity from ucm...')

        if self.datareader.__online():
            self.eurm = dot_product(self.urm[-10000, :], self.similarity_icm, k=500)
        else:
            self.eurm = dot_product(self.urm[test_playlists, :], self.similarity_icm, k=500)

        self.eurm = self.eurm.tocsr()
        return self.eurm

    def boost_eurm_genres(self, eurm, norm=None, gamma=0.01):
        test_pids = set(self.datareader.get_test_pids())

        data = []
        rows = []
        cols = []

        for genre in GENRES:
            for pid in self.genres_playlists[genre]:
                if pid in test_pids:
                    for track in self.genres[genre]:
                        if pid not in self.playlists_with_tracks:
                            data.append(1)
                            rows.append(self.test_playlists_eurm_idx[pid])
                            cols.append(track)
                        else:
                            if track not in self.known_tracks[pid]:
                                data.append(1)
                                rows.append(self.test_playlists_eurm_idx[pid])
                                cols.append(track)

        eurm_boosted = sparse.csr_matrix((data, (rows, cols)), shape=eurm.shape)

        return eurm + (eurm_boosted * gamma)


