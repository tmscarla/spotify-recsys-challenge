from personal.Tommaso.Recommenders.top_pop_rec import TopPopRecommender

from utils.datareader import Datareader
from utils.evaluator import Evaluator

"""
This script shows how to perform correctly an evaluation.
Basically you have to initialize an Evaluator object with csv files 
and then call the method evaluate which takes in input a numpy array
of shape (10.000, 500).
"""

# EVALUATOR
dr = Datareader(mode='offline', only_load=True)
ev = Evaluator(dr)

# TOP POP
t = TopPopRecommender()
t.fit(dr.get_df_train_interactions(), dr.get_df_test_interactions())
rec_list = t.make_recommendation(dr.get_df_test_playlists()['pid'].as_matrix(), remove_seen=True, is_submission=False)

# TOP POP FOLLOWERS
# urm = sparse.load_npz(ROOT_DIR + '/data/test1/matrices/urm.npz')
# urm.data = np.ones(len(urm.data))
# mpd_playlists = pd.read_csv(ROOT_DIR + '/data/original/train_playlists.csv', sep='\t')
#
# t = TopPopFollowersRecommender(mpd_playlists)
# t.fit(urm)
# rec_list = t.make_recommendation(test_playlists['pid'].as_matrix())

# TOP POP FILTER
# urm = sparse.load_npz(ROOT_DIR + '/data/test1/matrices/urm.npz')
# urm.data = np.ones(len(urm.data))
#
# t = TopPopFilterRecommender()
# t.fit(urm)
# rec_list = t.make_recommendation(test_playlists['pid'].as_matrix())

# EVALUATION ALL
# print(ev.evaluate(rec_list, 'top_pop_removed'))

# EVALUATION SINGLE METRIC
ev.evaluate(recommendation_list=rec_list, name='top_pop', verbose=True)


