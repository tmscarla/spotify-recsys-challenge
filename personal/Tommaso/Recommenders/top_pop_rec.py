import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.definitions import ROOT_DIR


class TopPopRecommender(object):
    """
    An unpersonalized top popular recommender. It recommends the top 500 tracks for each playlist.
    """

    def __init__(self):
        self.popular_tracks = []
        self.tracks_df = pd.read_csv(ROOT_DIR + '/data/original/tracks.csv', sep='\t')
        self.tracks_df.set_index(['tid'], inplace=True)
        self.challenge_playlists = pd.read_csv(ROOT_DIR + '/data/original/test_interactions.csv', sep='\t').groupby(
            ['pid'])['tid'].apply(list)

    def __str__(self):
        return "Top Pop Recommender"

    def fit(self, train_interactions, test_interactions):
        """
        Train the recommender with a list of known interactions playlist - track
        :param train_interactions: a Dataframe of interactions
        """

        print('Training Top Pop...')
        # Gather tracks of the same playlist and count their overall occurrences
        grouped_by_tid = train_interactions.groupby(['tid']).size().reset_index(name='counts')\
            .sort_values(['counts'], ascending=False)

        self.train_playlists = train_interactions.groupby(['pid'])['tid'].apply(list)
        self.test_playlists = test_interactions.groupby(['pid'])['tid'].apply(list)
        self.popular_tracks = np.squeeze(grouped_by_tid.as_matrix(columns=['tid']))[:1000]

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
            seen = self.test_playlists

        indices = np.array(seen.index.values)

        # Remove known tracks from the prediction
        if remove_seen and target_playlist in indices:
            hold_ix = ~np.in1d(self.popular_tracks, seen[target_playlist])
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
