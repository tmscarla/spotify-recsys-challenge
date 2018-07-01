from personal.Tommaso.Recommenders.top_pop_rec import TopPopRecommender
from utils.datareader import Datareader
from utils.submitter import Submitter

"""
This script shows how to perform correctly a submission.
Basically you have to initialize a Submitter object with csv files 
and then call the method submit which takes in input a numpy array
of recommendations of shape (10.000, 500).
"""

# SUBMITTER
dr = Datareader(mode='online', only_load=True)
sb = Submitter(dr)

# TOP POP
t = TopPopRecommender()
t.fit(dr.get_df_train_interactions(), dr.get_df_test_interactions())
rec_list = t.make_recommendation(dr.get_df_test_playlists()['pid'].as_matrix())

# SUBMISSION
# rec_list is an ordered list of recommendations
# This submission will be rejected due to duplicates occurrences.
sb.submit(recommendation_list=rec_list, name='top_pop', track='main', verify=True, gzipped=False)
