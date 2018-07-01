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
from utils.pre_processing import *
from utils.evaluator import Evaluator
from utils.post_processing import *
from utils.definitions import *
from utils.submitter import Submitter

datareader = Datareader(mode='online', only_load=True)
# ev = Evaluator(dr)


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def func():
    # Artists
    artists = list(datareader.get_df_artists()['artist_name'].as_matrix())
    artists = [str(x).lower() for x in artists]

    # Albums
    albums = list(datareader.get_df_test_albums()['album_name'].as_matrix())
    albums = [str(x).lower() for x in albums]

    # Playlist titles
    train_playlists_df = datareader.get_df_train_playlists()
    test_playlists_df = datareader.get_df_test_playlists()
    concat_df = pd.concat([train_playlists_df, test_playlists_df])

    if datareader.offline():
        concat_df = concat_df.sort_values(['pid'], ascending=True)

    playlists = concat_df['pid'].as_matrix()
    playlist_titles = concat_df['name'].as_matrix()
    playlist_titles = [str(x).lower() for x in playlist_titles]
    playlist_titles = np.array(playlist_titles)

    cat1 = np.array(datareader.get_test_pids_indices()).astype(np.int) + 1000000

    i = 0
    for title in playlist_titles[cat1]:
        for artist in artists:
            # if len(title) > 4:
            #     if title[0] in artist[0:2] or title[1] in artist[0:2]:
            #         d = levenshtein(title, artist)
            #         if d <= 1:
            if title == artist and len(title) > 3 and ' ' in title:
                i += 1

                print(title)
                #print(artist)
                #print('----------------')
    print(i)


func()
