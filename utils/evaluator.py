import pandas as pd
from utils.metrics import *
from tqdm import tqdm
import numpy as np
from scipy import sparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.definitions import ROOT_DIR
from utils.datareader import Datareader
import utils.post_processing as post


class Evaluator(object):

    def __init__(self, datareader):
        """
        Initialize the evaluator with dataframes.
        :param datareader: a Datareader object.
        """
        self.datareader = datareader

        self.tracks = datareader.get_df_tracks()
        self.test_playlists = datareader.get_df_test_playlists()
        self.eval_interactions = datareader.get_df_eval_interactions()

        # Create dict tracks-artists
        keys = list(self.tracks['tid'].as_matrix())
        values = list(self.tracks['arid'].as_matrix())
        self.dictionary = dict(zip(keys, values))

        # Group interactions for each playlist
        self.eval_tracks = self.eval_interactions.groupby(['pid'])['tid'].apply(list)
        self.eval_artists = self.eval_interactions.groupby(['pid'])['arid'].apply(list)

    def evaluate(self,
                 recommendation_list,
                 name,
                 verbose=True,
                 do_plot=False,
                 show_plot=False,
                 save=True,
                 return_result='mean',
                 old_mode=False):
        """
        Evaluate a list of recommendations according to precision, ndcg and clicks.
        Recommendations are assumed to be ordered according to the categories,
        i.e. the first 1000 belongs to the first category and so on.
        :param recommendation_list: a numpy array of (10.000, 500) predictions
        :param name: name of the test
        :param verbose: print what the algorithm is doing
        :param do_plot: if it false don't do any graphics
        :param show_plot: show the plot of the metrics with a pop up
        :param save: save the evaluation to a csv file
        :param return_result: ['mean', 'all'] If 'mean', return a sextuple containing the mean
               of each metric, if 'all' return the evaluation dataframe
        :param old_mode: if True, evaluate using the old version of the metrics
        """

        # Cumulative metrics scores for tracks
        cumulative_precision_t = 0.0
        cumulative_ndcg_t = 0.0
        cumulative_clicks_t = 0.0

        # Cumulative metrics scores for artists
        cumulative_precision_a = 0.0
        cumulative_ndcg_a = 0.0
        cumulative_clicks_a = 0.0

        # List of evaluation divided for each category and type
        evaluation_cat = {'precision_tracks': [], 'ndcg_tracks': [], 'clicks_tracks': [],
                          'precision_artists': [], 'ndcg_artists': [], 'clicks_artists': []}

        # Starting evaluation
        for i in tqdm(range(len(self.test_playlists)), desc='Evaluating', disable= not verbose):

            # Playlist identifier to test
            p = self.test_playlists['pid'][i]

            # Tracks
            recommended_tracks = np.array(recommendation_list[i])
            relevant_tracks = self.eval_tracks[p]

            # Artists
            recommended_artists = np.array([self.dictionary[t] for t in recommended_tracks])
            relevant_artists = self.eval_artists[p]

            # Old version
            if old_mode:
                # Tracks level
                cumulative_precision_t += r_precision_old(recommended_tracks, relevant_tracks)
                cumulative_ndcg_t += ndcg(recommended_tracks, relevant_tracks)
                cumulative_clicks_t += recommended_songs_clicks(recommended_tracks, relevant_tracks)

                # Artists level
                cumulative_precision_a += r_precision_old(recommended_artists, relevant_artists)
                cumulative_ndcg_a += ndcg(recommended_artists, relevant_artists)
                cumulative_clicks_a += recommended_songs_clicks(recommended_artists, relevant_artists)

            # New version
            else:
                # R-Precision
                precision_t, precision_a = r_precision(recommended_tracks, relevant_tracks,
                                                       recommended_artists, relevant_artists)
                cumulative_precision_t += precision_t
                cumulative_precision_a += precision_a

                # NDCG
                cumulative_ndcg_t += ndcg(recommended_tracks, relevant_tracks)
                cumulative_ndcg_a += ndcg(recommended_artists, relevant_artists)

                # Clicks
                cumulative_clicks_t += recommended_songs_clicks(recommended_tracks, relevant_tracks)
                cumulative_clicks_a += recommended_songs_clicks(recommended_artists, relevant_artists)

            # Compute categories separately
            if (i+1) % 1000 == 0:

                # Compute mean of playlists score for each category
                evaluation_cat['precision_tracks'].append(cumulative_precision_t / 1000)
                evaluation_cat['ndcg_tracks'].append(cumulative_ndcg_t / 1000)
                evaluation_cat['clicks_tracks'].append(cumulative_clicks_t / 1000)
                evaluation_cat['precision_artists'].append(cumulative_precision_a / 1000)
                evaluation_cat['ndcg_artists'].append(cumulative_ndcg_a / 1000)
                evaluation_cat['clicks_artists'].append(cumulative_clicks_a / 1000)

                # Reset cumulative scores
                cumulative_precision_t = 0.0
                cumulative_ndcg_t = 0.0
                cumulative_clicks_t = 0.0
                cumulative_precision_a = 0.0
                cumulative_ndcg_a = 0.0
                cumulative_clicks_a = 0.0

        # Compute mean
        evaluation_cat['precision_tracks'].append(np.mean(evaluation_cat['precision_tracks']))
        evaluation_cat['ndcg_tracks'].append(np.mean(evaluation_cat['ndcg_tracks']))
        evaluation_cat['clicks_tracks'].append(np.mean(evaluation_cat['clicks_tracks']))
        evaluation_cat['precision_artists'].append(np.mean(evaluation_cat['precision_artists']))
        evaluation_cat['ndcg_artists'].append(np.mean(evaluation_cat['ndcg_artists']))
        evaluation_cat['clicks_artists'].append(np.mean(evaluation_cat['clicks_artists']))

        evaluation_df = pd.DataFrame(data=evaluation_cat, index=['cat1', 'cat2', 'cat3', 'cat4', 'cat5',
                                                                 'cat6', 'cat7', 'cat8', 'cat9', 'cat10',
                                                                 'mean'])
        if verbose:
            print(evaluation_df, flush=True)

        # Plot the results
        if do_plot:
            self.plot_metrics('precision', evaluation_df, name, show_plot=show_plot)
            self.plot_metrics('ndcg', evaluation_df, name, show_plot=show_plot)
            self.plot_metrics('clicks', evaluation_df, name, show_plot=show_plot)

        # Save CSV file
        if save:
            evaluation_df.to_csv(ROOT_DIR + '/results/csv/test_' + name + '.csv', sep='\t')

        # Compute mean between categories
        overall_mean = (evaluation_cat['precision_tracks'][10],
                        evaluation_cat['ndcg_tracks'][10],
                        evaluation_cat['clicks_tracks'][10],
                        evaluation_cat['precision_artists'][10],
                        evaluation_cat['ndcg_artists'][10],
                        evaluation_cat['clicks_artists'][10])

        # Return results according to flags
        if return_result == 'mean':
            return overall_mean

        elif return_result == 'all':
            return overall_mean, evaluation_df

    def plot_metrics(self, metric, evaluation_df, name, bar_width=0.3, show_plot=True):
        """
        Plot the selected metric.
        :param metric: ['precision', 'ndcg', 'clicks']
        :param evaluation_df:
        :param name:
        :param bar_width:
        :return:
        """
        # Set positions
        tracks_pos = np.arange(1, 11)
        artists_pos = [x + bar_width for x in tracks_pos]

        fig, ax = plt.subplots()
        ax.grid(alpha=0.7)

        tracks_values = evaluation_df[metric + '_tracks'].as_matrix()
        artists_values = evaluation_df[metric + '_artists'].as_matrix()

        tracks_rects = ax.bar(tracks_pos,
                tracks_values[:-1],
                width=bar_width,
                color='black',
                label='tracks')
        artists_rects = ax.bar(artists_pos,
                artists_values[:-1],
                width=bar_width,
                color='white',
                edgecolor='black',
                label='artists')

        plt.title(name +
                  '\n' + metric + '\n'
                  'tracks = ' + str(tracks_values[-1]) +
                  '\nartists = ' + str(artists_values[-1]))

        plt.xlabel("Categories")
        plt.grid()
        plt.legend()
        plt.xticks(np.arange(1, 11), list(filter(lambda x: "cat" + str(x), np.arange(1, 11))))

        # Set limits
        axes = plt.gca()

        if metric == 'clicks':
            axes.set_ylim([0, 51])
        else:
            axes.set_ylim([0, 1])

        plt.subplots_adjust(left=0.06, bottom=0.10, right=0.98, top=0.82,
                              wspace=None, hspace=None)

        self.autolabel(ax, tracks_rects)
        self.autolabel(ax, artists_rects)

        fig.savefig(ROOT_DIR + '/results/img/' + name + '_' + metric + '.png')

        if show_plot:
            plt.show()

        plt.close(fig)

    def autolabel(self, ax, rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    1.05 * height,
                    '%.2f' % height,
                    fontsize=8,
                    ha='center', va='bottom')

    def evaluate_single_metric(self, recommendation_list, name, metric, level, cat='mean', verbose=True):
        """
        Evaluate a single metric on track or artist level.
        :param recommendation_list: a numpy array of (10.000, 500) predictions
        :param name: name of the test
        :param metric: ['prec', 'ndcg', 'clicks', 'sum'] where sum = ndcg + r_prec
        :param level: ['track', 'artist']
        :param name: name of the test
        :param cat: ['mean', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        :return: score: the score for the selected metric for one or all the categories
        """

        # Check arguments
        metrics = ['prec', 'ndcg', 'clicks','sum']
        assert metric in metrics
        levels = ['track', 'artist']
        assert level in levels
        categories = ['mean', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert cat in categories

        # Cumulative metric scores
        cumulative_score = 0.0

        # Single category
        if cat in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

            # Tracks evaluation
            if level == 'track':
                for i in tqdm(range(1000), desc='Evaluating ' + metric + ' at track level', disable=not verbose):
                    # Playlist identifier to test
                    p = self.test_playlists['pid'][i + (1000 * (int(cat) - 1))]

                    # Recommendation
                    recommended_tracks = recommendation_list[i + (1000 * (int(cat) - 1))]
                    relevant_tracks = self.eval_tracks[p]
                    recommended_artists = np.array([self.dictionary[t] for t in recommended_tracks])
                    relevant_artists = self.eval_artists[p]

                    if metric == 'prec':
                        cumulative_score += r_precision(recommended_tracks, relevant_tracks,
                                                        recommended_artists, relevant_artists)[0]
                    elif metric == 'ndcg':
                        cumulative_score += ndcg(recommended_tracks, relevant_tracks)
                    elif metric == 'clicks':
                        cumulative_score += recommended_songs_clicks(recommended_tracks, relevant_tracks)
                    elif metric == 'sum':
                        cumulative_score += r_precision(recommended_tracks, relevant_tracks,
                                                        recommended_artists, relevant_artists)[0]\
                                            + ndcg(recommended_tracks, relevant_tracks)

                score = cumulative_score / 1000

            # Artists evaluation
            elif level == 'artist':
                for i in tqdm(range(len(self.test_playlists)), desc='Evaluating ' + metric + ' at artist level', disable=not verbose):
                    # Playlist identifier to test
                    p = self.test_playlists['pid'][i + (1000 * (int(cat) - 1))]

                    # Recommendation
                    recommended_tracks = recommendation_list[i + (1000 * (int(cat) - 1))]
                    relevant_tracks = self.eval_tracks[p]
                    recommended_artists = np.array([self.dictionary[t] for t in recommended_tracks])
                    relevant_artists = self.eval_artists[p]

                    if metric == 'prec':
                        cumulative_score += r_precision(recommended_tracks, relevant_tracks,
                                                        recommended_artists, relevant_artists)[1]
                    elif metric == 'ndcg':
                        cumulative_score += ndcg(recommended_artists, relevant_artists)
                    elif metric == 'clicks':
                        cumulative_score += recommended_songs_clicks(recommended_artists, relevant_artists)

                score = cumulative_score / 1000

            if verbose:
                print(name)
                print('cat =', cat)
                print('level =', level)
                print(metric, '=', score)

            return score

        # All categories
        elif cat == 'mean':

            # Tracks evaluation
            if level == 'track':
                for i in tqdm(range(len(self.test_playlists)), desc='Evaluating ' + metric + ' at track level', disable=not verbose):
                    # Playlist identifier to test
                    p = self.test_playlists['pid'][i]

                    # Recommendation
                    recommended_tracks = recommendation_list[i]
                    relevant_tracks = self.eval_tracks[p]
                    recommended_artists = np.array([self.dictionary[t] for t in recommended_tracks])
                    relevant_artists = self.eval_artists[p]

                    if metric == 'prec':
                        cumulative_score += r_precision(recommended_tracks, relevant_tracks,
                                                        recommended_artists, relevant_artists)[0]
                    elif metric == 'ndcg':
                        cumulative_score += ndcg(recommended_tracks, relevant_tracks)
                    elif metric == 'clicks':
                        cumulative_score += recommended_songs_clicks(recommended_tracks, relevant_tracks)

                score = cumulative_score / len(self.test_playlists)

            # Artists evaluation
            elif level == 'artists':
                for i in tqdm(range(len(self.test_playlists)), desc='Evaluating ' + metric + ' at artist level', disable= not verbose):
                    # Playlist identifier to test
                    p = self.test_playlists['pid'][i]

                    # Recommendation
                    recommended_tracks = recommendation_list[i]
                    relevant_tracks = self.eval_tracks[p]
                    recommended_artists = np.array([self.dictionary[t] for t in recommended_tracks])
                    relevant_artists = self.eval_artists[p]

                    if metric == 'prec':
                        cumulative_score += r_precision(recommended_tracks, relevant_tracks,
                                                        recommended_artists, relevant_artists)[1]
                    elif metric == 'ndcg':
                        cumulative_score += ndcg(recommended_artists, relevant_artists)
                    elif metric == 'clicks':
                        cumulative_score += recommended_songs_clicks(recommended_artists, relevant_artists)

                score = cumulative_score / len(self.test_playlists)

            if verbose:
                print(name)
                print('cat =', cat)
                print('level =', level)
                print(metric, '=', score)

            return score

    def evaluate_single_playlists(self, recommendation_list, indices=[], verbose=True):
        """
        Evaluate singularly a list of playlist within the recommendation list.
        If only one playlist is evaluated, the function returns the result in form of a sextuple.
        :param recommendation_list: a numpy array of (10.000, 500) predictions
        :param indices: indices of the playlists to evaluate wrt the recommendation_list
        :param verbose: print the result of each playlist
        :return result: a sextuple (prec_t, ndcg_t, clicks_t, prec_a, ndcg_a, clicks_a)
        """

        pid_to_name = self.datareader.get_pid_to_name_dict()

        for i in indices:
            # Playlist identifier to test
            p = self.test_playlists['pid'][i]

            # Recommendation
            recommended_tracks = recommendation_list[i]
            recommended_artists = np.array([self.dictionary[t] for t in recommended_tracks])

            # Test
            relevant_tracks = self.eval_tracks[p]
            relevant_artists = self.eval_artists[p]

            # R-Precision
            prec_t, prec_a = r_precision(recommended_tracks, relevant_tracks,
                                                   recommended_artists, relevant_artists)

            # NDCG
            ndcg_t = ndcg(recommended_tracks, relevant_tracks)
            ndcg_a = ndcg(recommended_artists, relevant_artists)

            # Clicks
            clicks_t = recommended_songs_clicks(recommended_tracks, relevant_tracks)
            clicks_a = recommended_songs_clicks(recommended_artists, relevant_artists)

            # Print metrics
            if verbose:
                print('------------------------------------------------')
                print('INDEX =', i, '|', 'PID =', p, '|', 'NAME =', pid_to_name[p])
                print('------------------------------------------------')
                print('prec_t =', prec_t)
                print('ndcg_t =', ndcg_t)
                print('clicks_t =', clicks_t)
                print('------------------------------------------------')
                print('prec_a =', prec_a)
                print('ndcg_a =', ndcg_a)
                print('clicks_a =', clicks_a)
                print('------------------------------------------------')

            # Return result if only one playlist is evaluated
            if len(indices) == 1:
                return prec_t, ndcg_t, clicks_t, prec_a, ndcg_a, clicks_a

    def fast_evaluate_eurm(self, eurm,
                           target_pids=None,
                           name='fast_evaluation',
                           verbose=True,
                           do_plot=False,
                           show_plot=False,
                           save=False,
                           return_result='all'):
        """
        Directly evaluate a eurm of shape (10K, 2.2M) removing seed tracks and converting it
        into a recommendation list.
        """
        if target_pids is None:
            target_pids = self.datareader.get_test_pids()
        eurm = sparse.csr_matrix(eurm[target_pids])
        #eurm = post.eurm_remove_seed(eurm,  datareader=self.datareader) no needed, now it's done in the next function
        rec_list = post.eurm_to_recommendation_list(eurm, verbose=False, datareader=self.datareader, remove_seed=True)
        result = self.evaluate(rec_list, name, verbose=verbose, return_result=return_result,
                               do_plot=do_plot, show_plot=show_plot, save=save)
        return result
