import sys
from scipy import sparse as sps
import utils.pre_processing as pre
from boosts.hole_boost import HoleBoost
from boosts.match_boost import MatchBoost
from boosts.tail_boost import TailBoost
from boosts.album_boost import AlbumBoost
from boosts.top_boost import TopBoost
from utils.post_processing import *
from utils.pre_processing import *
from utils.submitter import Submitter
from utils.ensembler import *
from boosts.generate_similarity import generate_similarity


# INIT
dr = Datareader(mode='online', only_load=True, verbose=False)
sb = Submitter(dr)

####### LOAD MATRICES AFTER BAYESIAN OPTIMIZATION  #####################################

cluster1 = sps.load_npz(ROOT_DIR + '/final_npz_main/ensembled_ar1_online.npz')
cluster2 = sps.load_npz(ROOT_DIR + '/final_npz_main/ensembled_ar2_online.npz')
cluster3 = sps.load_npz(ROOT_DIR + '/final_npz_main/ensembled_ar3_online.npz')
cluster4 = sps.load_npz(ROOT_DIR + '/fin©al_npz_main/ensembled_ar4_online.npz')

clustered_approach_online = cluster1 + cluster2 + cluster3 + cluster4

ensembled1 = sps.load_npz(ROOT_DIR + '/final_npz_main/ensembled_MAIN_online_half1.npz')
ensembled2 = sps.load_npz(ROOT_DIR + '/final_npz_main/ensembled_MAIN_online_half2.npz')

ensembled = ensembled1 + ensembled2

####### POSTPROCESSING #################################################################

# COMBINE
eurm_ens = combine_two_eurms(clustered_approach_online, ensembled, cat_first=[4, 5, 6, 8, 10])
sim = generate_similarity('online')

# HOLEBOOST
hb = HoleBoost(similarity=sim, eurm=eurm_ens, datareader=dr, norm=norm_l1_row)
eurm_ens = hb.boost_eurm(categories=[8], k=300, gamma=1)
hb = HoleBoost(similarity=sim, eurm=eurm_ens, datareader=dr, norm=norm_l1_row)
eurm_ens = hb.boost_eurm(categories=[10], k=150, gamma=1)

# TAILBOOST
tb = TailBoost(similarity=sim, eurm=eurm_ens, datareader=dr, norm=norm_l2_row)
eurm_ens = tb.boost_eurm(categories=[9, 7, 6, 5],
                         last_tracks=[10, 3, 3, 3],
                         k=[100, 80, 100, 100],
                         gamma=[0.01, 0.01, 0.01, 0.01])

# ALBUMBOOST
ab = AlbumBoost(dr, eurm_ens)
eurm_ens = ab.boost_eurm(categories=[3, 4, 7, 9], gamma=2, top_k=[3, 3, 10, 40])

# SUBMISSION
rec_list = eurm_to_recommendation_list_submission(eurm_ens, datareader=dr)
sb.submit(rec_list, name='main_track', track='main')
