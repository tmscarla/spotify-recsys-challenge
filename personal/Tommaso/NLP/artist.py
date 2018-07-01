import pandas as pd
import numpy as np
import nltk
from nltk import stem
from nltk.tokenize import RegexpTokenizer
from utils.datareader import Datareader
from tqdm import tqdm
from scipy import sparse
from difflib import SequenceMatcher
from difflib import get_close_matches



class ArtistToken(object):

    def __init__(self, datareader, stopwords=[], verbose=True):
        """
        :param datareader: a Datareader object
        :param stopwords: a list of stopwords
        :param verbose: set verbosity
        """

        self.datareader = datareader
        self.stopwords = stopwords

        # Read dataframes and put together test and train
        self.artists = list(self.datareader.get_df_artists()['artist_name'].as_matrix())
        self.artists = [str(x).lower() for x in self.artists]

        print(self.artists)

        self.artists_dict = dict(zip(self.artists, list(np.arange(len(self.artists)))))

        train_playlists_df = self.datareader.get_df_train_playlists()
        test_playlists_df = self.datareader.get_df_test_playlists()
        concat_df = pd.concat([train_playlists_df, test_playlists_df])
        concat_df = concat_df.sort_values(['pid'], ascending=True)

        self.playlist_titles = concat_df['name'].as_matrix()

        self.dictionary = dict()

        self.tokens_dict = dict()

        self.__set_tokens()
        self.__set_params_()

    def __set_tokens(self):

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

                        if s in self.tokens_dict.keys():
                            self.tokens_dict[s].add(i)
                        else:
                            self.tokens_dict[s] = {i}

    def __set_params_(self):

        for token in tqdm(self.tokens_dict.keys(), desc='Playlist titles matching'):

            if token not in self.stopwords and len(token) > 1:
                artists = get_close_matches(token, self.artists, cutoff=1.0, n=1)
                #artists = process.extract(token, self.artists, limit=1)

                for a in artists:
                    arid = self.artists_dict[a]

                    if arid in self.dictionary.keys():
                        self.dictionary[arid].union(self.tokens_dict[token])
                    else:
                        self.dictionary[arid] = set().union(self.tokens_dict[token])

    def get_playlist_artist_matrix(self):
        rows = []
        cols = []
        data = []

        for arid in tqdm(self.dictionary.keys(), desc='Build (playlists, artists) matrix'):

            for playlist in self.dictionary[arid]:
                rows.append(playlist)
                cols.append(arid)
                data.append(1)

        self.playlist_artist_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.playlist_titles),
                                                                      len(self.artists)))

        return self.playlist_artist_matrix


if __name__ == '__main__':
    n = ArtistToken(Datareader(mode='offline', only_load=True, verbose=False))
    m = n.get_playlist_artist_matrix()
    sparse.save_npz('pam.npz', m)
