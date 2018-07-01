from utils.audio_feature_cluster import *
import pandas as pd


if __name__ == '__main__':
    # Give the features
    feats = ['duration_ms']

    # Give the number of desired clusters
    k = 4

    # Load ['tid', 'feature'] df
    tracks = pd.read_csv('../data/original/tracks.csv', sep='\t')

    # Cluster result would be a df ['tid', 'feat_cluster']
    df_c = tracks[['tid']]
    for feature in feats:
        df_feat = tracks[['tid',feature]]
        feat_c = cluster(df=df_feat, K=k, feat=feature, verbose=True)
        # clean
        feat_c = feat_c[['tid', feature+'_cluster']]
        # merge
        df_c = df_c.merge(feat_c, left_on='tid', right_on='tid', how='inner')
        # check
        print('Check merge:',feature, len(df_c), len(feat_c),'should be equal.')

    print(df_c.head())
    # dump data
    #df_c.to_csv('../data/enriched/tracks_audio_features_clustered(K='+str(k)+').csv', index=False, sep='\t')