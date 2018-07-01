from utils.audio_feature_cluster import *
import pandas as pd
import numpy  as np
from tqdm import tqdm
import scipy.sparse as sp
from utils.definitions import ROOT_DIR
from utils.datareader import Datareader

"""
This file is used to generate the layered item content matrix and user rating matrix 
for creative track recommendation.
"""

######################################################################################################
## Audio feature cluster
######################################################################################################
# Number of clusters
k = 4

# Load ['tid', 'feature'] df
tracks = pd.read_csv(ROOT_DIR+ '/data/enriched/tracks_v4.0.csv', sep='\t')

# Give the features
feats = ['acousticness', 'danceability', 'duration_ms', 'energy',
         'instrumentalness',
         'liveness', 'loudness', 'speechiness', 'tempo', 'valence','popularity']

# for float features Cluster result would be a df ['tid', 'feat_cluster'] 
df_c = tracks[['tid']]
for feature in tqdm(feats, desc= 'Clustering float features'):
    df_feat = tracks[['tid',feature]]
    feat_c = cluster(df=df_feat, K=k, feat=feature, verbose=False)
    # clean
    feat_c = feat_c[['tid', feature+'_cluster']]
    # merge
    df_c = df_c.merge(feat_c, left_on='tid', right_on='tid', how='inner')
    
# Dump cluster result
df_c.to_csv(ROOT_DIR+'/data/enriched/tracks_audio_features_clustered(K='+str(k)+').csv', index=False, sep='\t')


######################################################################################################
## Generate layered arid with audio features
###################################################################################################### 
mode = 'all_line'
train_intr = pd.read_csv(ROOT_DIR+'/data/original/tracks.csv', sep='\t', usecols=['tid', 'arid'], dtype={'arid':str})

feats = ['acousticness_cluster', 'danceability_cluster', 'duration_ms_cluster', 'energy_cluster',
         'instrumentalness_cluster',
         'liveness_cluster', 'loudness_cluster', 'speechiness_cluster', 'tempo_cluster', 'valence_cluster','popularity_cluster']
for feat in tqdm(feats, desc = 'Generating layered artist with audio features'):
    af = pd.read_csv(ROOT_DIR+'/data/enriched/tracks_audio_features_clustered(K='+str(k)+').csv', sep='\t',usecols=['tid', feat], dtype={feat: str})
    # merge
    df = train_intr.merge(af, left_on='tid', right_on='tid', how='inner')
    del af
    # refine
    df['arid_'+feat] = df['arid'].astype(str).str.cat(df[feat].astype(str), sep='-')
    # prepare unique ids for 'arid_feat'
    df_tmp = pd.DataFrame()
    df_tmp['arid_'+feat] = df['arid_'+feat]
    df_tmp = df_tmp.drop_duplicates()
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp['new_arid'] = df_tmp.index
    # attact new_arid to tid
    df = df.merge(df_tmp, left_on='arid_'+feat, right_on='arid_'+feat, how='inner')
    del df_tmp
    # start building icm
    n_tracks = 2262292
    trs = df['tid'].values
    n = len(df)
    arids = df['new_arid'].values
    del df
    n_arids = arids.max() + 1  # index starts from 0
    # create partial icm 
    icm_ar = sp.csr_matrix((np.ones(n), (trs, arids)), shape=(n_tracks, n_arids),
                           dtype=np.int32)
    # dump icm
    sp.save_npz(ROOT_DIR+'/data/enriched/tmp_icms/'+mode+'_refine_arid_'+feat + ".npz", icm_ar)


######################################################################################################
## Generate layered pid with audio features
###################################################################################################### 
mode = 'online'
feats = ['acousticness_cluster', 'danceability_cluster', 'duration_ms_cluster', 'energy_cluster',
         'instrumentalness_cluster', 'liveness_cluster', 'loudness_cluster', 'speechiness_cluster', 
         'tempo_cluster', 'valence_cluster','popularity_cluster' ]
for feat in tqdm(feats, desc='Generating layered playlists with audio features'):
    af = pd.read_csv(ROOT_DIR+'/data/enriched/tracks_audio_features_clustered(K=' + str(k) + ').csv', sep='\t',
                         usecols=['tid', feat],
                         dtype={feat: str})  
    if mode == 'offline':
        tracks = pd.read_csv(ROOT_DIR + '/data/test1/train_interactions.csv', sep='\t', dtype={'pid': str})
    if mode == 'online':
        tracks = pd.read_csv(ROOT_DIR + '/data/original/train_interactions.csv', sep='\t', dtype={'pid': str})
    # clean
    tracks = tracks[['tid', 'pid']]
    # merge
    tracks = tracks.merge(af, left_on='tid', right_on='tid', how='inner')
    del af
    # refine
    tracks['pid_genre'] = tracks['pid'].str.cat(tracks[feat], sep='-')
    # prepare unique ids for 'pid_genre'
    df_tmp = pd.DataFrame()
    df_tmp['pid_genre'] = tracks['pid_genre']
    df_tmp = df_tmp.drop_duplicates()
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp['pidgenid'] = df_tmp.index
    # attact pidgenid to tid
    tracks = tracks.merge(df_tmp, left_on='pid_genre', right_on='pid_genre', how='inner')
    del df_tmp
    # start building icm
    n_tracks = 2262292
    trs = tracks['tid'].values
    n = len(tracks)
    pidgenids = tracks['pidgenid'].values
    del tracks
    n_pidgenids = pidgenids.max() + 1  # index starts from 0
    # create partial icm 
    icm_ = sp.csr_matrix((np.ones(n), (trs, pidgenids)), shape=(n_tracks, n_pidgenids),
                           dtype=np.int32)
    # dump icm
    sp.save_npz(ROOT_DIR + '/data/enriched/tmp_icms/'+mode+'_refine_pid_'+feat + ".npz", icm_)


######################################################################################################
## Generate layered pid with artist and album
###################################################################################################### 
mode = 'online'
dr = Datareader(mode=mode, only_load=True, verbose=False)
train_intr = dr.get_df_train_interactions()
feats = ['alid', 'arid']
for feat in tqdm(feats, desc='Generating layered playlist'):
    af = pd.read_csv(ROOT_DIR + '/data/original/tracks.csv', sep='\t',usecols=['tid', feat], dtype={feat: str})
    # merge
    df = train_intr.merge(af, left_on='tid', right_on='tid', how='inner')
    del af
    # refine
    df['pid_'+feat] = df['pid'].astype(str).str.cat(df[feat].astype(str), sep='-')
    # prepare unique ids for 'pid_feat'
    df_tmp = pd.DataFrame()
    df_tmp['pid_'+feat] = df['pid_'+feat]
    df_tmp = df_tmp.drop_duplicates()
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp['new_pid'] = df_tmp.index
    # attact new_pid to tid
    df = df.merge(df_tmp, left_on='pid_'+feat, right_on='pid_'+feat, how='inner')
    del df_tmp
    # start building icm
    n_tracks = 2262292
    trs = df['tid'].values
    n = len(df)
    pids = df['new_pid'].values
    del df
    n_pids = pids.max() + 1  # index starts from 0
    # create partial icm 
    icm_ = sp.csr_matrix((np.ones(n), (trs, pids)), shape=(n_tracks, n_pids),
                           dtype=np.int32)
    # dump icm
    sp.save_npz(ROOT_DIR + '/data/enriched/tmp_icms/'+mode+'_refine_pid_'+feat + ".npz", icm_)
