import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.definitions import ROOT_DIR
from sklearn.utils.sparsefuncs import inplace_row_scale
from sklearn.preprocessing import normalize


class TopPopFollowersRecommender(object):
    """
    An unpersonalized top popular recommender. It recommends the top 500 tracks for each playlist.
    """

    def __init__(self, train_playlists_df):
        self.popular_tracks = []
        self.tracks_df = pd.read_csv(ROOT_DIR + '/data/original/tracks.csv', sep='\t')
        self.tracks_df.set_index(['tid'], inplace=True)
        self.challenge_playlists = pd.read_csv(ROOT_DIR + '/data/original/test_interactions.csv', sep='\t').groupby(
            ['pid'])['tid'].apply(list)

        self.followers = train_playlists_df['num_followers'].as_matrix()

    def __str__(self):
        return "Top Pop Followers Recommender"

    def fit(self, urm):
        """
        Train the recommender with a list of known interactions playlist - track
        :param urm: the user rating matrix
        """
        print('Training Top Pop Followers...')
        self.urm = urm

        # Remove duplicates
        self.urm.data = np.ones(len(self.urm.data))

        # Normalize
        self.followers = normalize(self.followers.reshape(self.followers.shape[0],-1),
                                   norm='l2', axis=0).reshape(self.followers.shape)

        # Scale urm according to followers
        inplace_row_scale(urm, self.followers)

        self.track_weighted = urm.sum(axis=0)
        self.track_weighted = np.squeeze(np.asarray(self.track_weighted))

        self.popular_tracks = np.argsort(self.track_weighted)[::-1][:10000]

    def recommend(self, target_playlist, remove_seen=True, is_submission=False):
        """
        Compute a single recommendation for a target playlist.
        :param target_playlist: the pid of the target playlist
        :param remove_seen: if true, tracks already present in the target_playlist are removed
        :param is_submission: if true, returns a prediction with uris instead of indices
        :return: recommended_tracks or recommended_tracks_uri
        """

        # Determine the known tracks
        if is_submission:
            seen = self.challenge_playlists[target_playlist]
        else:
            seen = self.urm[target_playlist].indices

        # Remove known tracks from the prediction
        if remove_seen:
            hold_ix = ~np.in1d(self.popular_tracks, seen)
            recommended_tracks = self.popular_tracks[hold_ix]
            recommended_tracks = recommended_tracks[0:500]
        else:
            recommended_tracks = self.popular_tracks[0:500]

        # Return tids or uris
        if is_submission:
            recommended_tracks_uri = [self.tracks_df['track_uri'][t] for t in recommended_tracks]
            return recommended_tracks_uri
        else:
            return recommended_tracks

    def make_recommendation(self, target_playlists, remove_seen=True, is_submission=False):
        """
        Produce a recommendation in the standard form.
        :param target_playlists: a list of playlists to make recommendation for
        :param remove_seen: if true, tracks already present in the target_playlist are removed
        :param is_submission: if true, returns a prediction with uris instead of indices
        :return: recommendation_list: a numpy array of arrays of recommendation of shape = (10000, 500)
        """

        recommendation_list = []

        for i in tqdm(range(len(target_playlists))):
            recommendation_list.append(self.recommend(target_playlists[i], remove_seen, is_submission))

        return np.array(recommendation_list)
