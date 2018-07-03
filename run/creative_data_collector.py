import spotipy
import spotipy.oauth2
from tqdm import tqdm
from utils.definitions import ROOT_DIR
import pandas as pd
import numpy as np
import sys

"""
This file is used to download audio features and track popularity from Spotify API, using
Spotipy, which is a lightweight Python library for the Spotify Web API.

In order to use it, you have to provide the client_id and client_secret.
More details can be found here:
https://developer.spotify.com/documentation/web-api/quick-start/
http://spotipy.readthedocs.io/en/latest/
"""


if len(sys.argv) < 3:
        print('Usage: python creative_data_collector.py <clientID> <clientSecret>')
        exit()
arg = sys.argv[1:]
clientID = arg[0]
clientSecret = arg[1]

# Prepare spotipy api
credentials = spotipy.oauth2.SpotifyClientCredentials(client_id=clientID,
                                                     client_secret=clientSecret)
spotify = spotipy.Spotify(client_credentials_manager=credentials)

# Load targht tracks
tracks = pd.read_csv(ROOT_DIR + '/data/original/tracks.csv', sep='\t')
uris = tracks[['track_uri']]

# Download audio features and track popularity
audioDict = {}
for uri in tqdm(uris.track_uri, desc='Downloading audio features'):
    audioDict[uri] = spotify.audio_features(uri)
popDict = {}
for uri in tqdm(uris.track_uri, desc='Downloading track popularity'):
    try:
        popDict[uri] = spotify.track(uri)['popularity']
        print(popDict[uri]['popularity'] )
    except:
        pass
    
    
# Build dataframe for audio feats
feats = ['acousticness','danceability','energy','instrumentalness',
             'liveness','loudness','speechiness','tempo','valence','key','mode',
             'time_signature']
uriList = []
featLists = {}
for feat in feats:
    featLists[feat] = []
for key in tqdm(audioDict.keys(), desc='Building dataframe'):
    uri = key
    audio = audioDict[key]
    if audio[0]:
        uriList.append(key)
        feat_dict = audio[0]
        for feat in feats:
            featLists[feat].append(feat_dict[feat])
    else:
        pass
df = pd.DataFrame()
df['track_uri'] = uriList
for feat in feats:
    df[feat] = featLists[feat]
    
# Merge info
tracks = tracks.merge(df, left_on='track_uri', right_on='track_uri', how='left')

# Build dataframe for popularity
uriList = []
popList = []
for key in popDict.keys():
    uriList.append(key)
    popList.append(popDict[key])
df = pd.DataFrame()
df['track_uri'] = uriList
df['popularity'] = popList

# Merge info
tracks = tracks.merge(df, left_on='track_uri', right_on='track_uri', how='left')

# Fill missing value
tracks = tracks.fillna(0)

# Dump dataframe
print('Saving data to "data/enriched/tracks_v4.0.csv"')
tracks.to_csv(ROOT_DIR + '/data/enriched/tracks_v4.0.csv', sep='\t', index=False)
