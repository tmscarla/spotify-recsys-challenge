import numpy as np
import pandas as pd
from tqdm import tqdm

class RandomRecommender(object):
    """
    A random recommender. It recommends 500 random tracks for each playlist.
    """

    def __init__(self, tracks, challenge_playlists):
        self.random_tracks = []
        self.tracks_df = tracks
        #self.challenge_playlists = challenge_playlists.groupby(['pid'])['tid'].apply(list)

    def __str__(self):
        return "Random Recommender"

    def fit(self, train_interactions):
        self.random_tracks = np.random.choice(len(self.tracks_df), 1000)
        self.train_playlists = train_interactions.groupby(['pid'])['tid'].apply(list)

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
            seen = self.challenge_playlists
        else:
            seen = self.train_playlists

        # Remove known tracks from the prediction
        if remove_seen and target_playlist in seen.index:
            hold_ix = ~np.in1d(self.random_tracks, seen[target_playlist])
            recommended_tracks = self.random_tracks[hold_ix]
            recommended_tracks = recommended_tracks[0:500]
        else:
            recommended_tracks = self.random_tracks[0:500]

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

        recommendation_list = np.empty([len(target_playlists), 500])

        for i in tqdm(range(len(target_playlists))):
            recommendation_list[i] = self.recommend(target_playlists[i], remove_seen, is_submission)

        return recommendation_list

