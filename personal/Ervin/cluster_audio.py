import numpy as np
import scipy.sparse as sps
import pandas as pd
from numpy.core.operand_flag_tests import inplace_add
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from utils.datareader import Datareader

def cluster_audio(n_clusters=40):
    scaler = MinMaxScaler()

    print('[Reading tracks files]')
    tracks = pd.read_csv('../../data/original/tracks.csv', sep='\t')
    tracks = tracks[['track_uri','tid']]

    print('[Reading audio features file]')
    track_features = pd.read_csv('../../data/v2.0/tracks_audio_features_v2.0.csv', sep='\t')
    track_features['valence'].fillna(track_features['valence'].mean(), inplace=True)
    track_uri = track_features['uri']
    track_features = track_features.drop(columns=['duration_ms', 'uri', 'key', 'time_signature'])
    track_features = track_features.astype(np.float32)
    track_features['track_uri'] = track_uri.values
    track_features['tempo'] = scaler.fit_transform(track_features.tempo.values.reshape(-1, 1))
    track_features['loudness'] = scaler.fit_transform(track_features.loudness.values.reshape(-1, 1))

    print('[Fitting KMeans with {} clusters]'.format(n_clusters))
    model = KMeans(n_clusters=n_clusters, n_jobs=-1, n_init=10)
    track_features['cluster'] = model.fit_predict(track_features[track_features.columns[:-1]])
    track_features = track_features[['track_uri', 'cluster']]
    track_genres = pd.merge(tracks, track_features, on='track_uri', how='left', left_index=True)
    track_genres.dropna(how='any', inplace=True)

    print('[Saving clusters]')
    np.savez_compressed('../../data/v2.0/genre_clusters_'+str(n_clusters),
                        genres=track_genres.cluster.values, tid=track_genres.tid.values)

if __name__ == '__main__':
    cluster_audio()