import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.definitions import ROOT_DIR
import sys
from utils.datareader import Datareader

"""
This file has the only purpose to extract from the challenge set json file a list
of useful csv in order to make submissions in the right format.

python challenge_set_to_csv.py path/to/challenge_set.json
"""


def convert(path):
    # LOAD DATA
    data = json.load(open(ROOT_DIR + "/data/challenge/challenge_set.json"))
    dr = Datareader(mode='online', only_load=True, verbose=False)

    # CHALLENGE PLAYLISTS
    target_playlists = data['playlists']

    # Drop tracks and reorder
    target_playlists_df = pd.DataFrame(target_playlists)
    target_playlists_df = target_playlists_df.drop(['tracks'], axis=1)
    target_playlists_df.sort_values(by=['pid'], inplace=True)

    # Set pid as first column
    cols = target_playlists_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    target_playlists_df = target_playlists_df[cols]

    # Save csv file
    target_playlists_df.to_csv('test_playlists.csv', sep='\t', index=False)

    pids = []
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        pids.extend(dr.get_test_pids(cat=i))
    test_playlists_df = pd.read_csv('test_playlists.csv', sep='\t', encoding='utf-8')

    test_playlists_df = test_playlists_df.set_index(['pid'])

    # Load and resave csv file ordered by cat
    test_playlists_df = test_playlists_df.reindex(pids)
    test_playlists_df['pid'] = test_playlists_df.index
    test_playlists_df.to_csv('test_playlists.csv', sep='\t', index=False, encoding='utf-8')

    # Dict uri -> tid
    tracks_df = dr.get_df_tracks()

    # Create dict track_uri - track_id
    values = list(tracks_df['tid'].as_matrix())
    keys = list(tracks_df['track_uri'].as_matrix())
    uri_to_tid = dict(zip(keys, values))

    # CHALLENGE INTERACTIONS
    iteractions = [[],[],[]]
    for p in tqdm(range(len(target_playlists))):
        tracks = data["playlists"][p]["tracks"]
        playlistId = data["playlists"][p]["pid"]
        iteractions[0].extend([playlistId] * len(tracks))

        for t in range(len(tracks)):
            tid = uri_to_tid[tracks[t]["track_uri"]]
            iteractions[1].extend([tid])
            iteractions[2].extend([tracks[t]["pos"]])
    d = {'pid': iteractions[0], 'tid': iteractions[1], 'pos': iteractions[2]}
    all_interactions = pd.DataFrame(d)
    all_interactions.sort_values(by=['pid'], inplace=True)

    all_interactions.to_csv('test_interactions.csv', sep='\t', index=False)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Please provide path for challenge_set.json file.')
    else:
        path = sys.argv[1]
        convert(path)
        print('Saved data to data/original')
