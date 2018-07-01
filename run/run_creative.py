import sys
from scipy import sparse as sps
import utils.pre_processing as pre
from boosts.hole_boost import HoleBoost
from boosts.match_boost import MatchBoost
from boosts.tail_boost import TailBoost
from boosts.album_boost import AlbumBoost
from boosts.top_boost import TopBoost
from utils.evaluator import Evaluator
from utils.post_processing import *
from utils.pre_processing import *
from utils.submitter import Submitter
from utils.ensembler import *
from boosts.generate_similarity import generate_similarity



####### PREPROCESSING ##################################################################

# INIT
dr = Datareader(mode='online', only_load=True, verbose=False)
sb = Submitter(dr)


####### RECOMMENDATION #################################################################


####### POSTPROCESSING #################################################################

# LOAD AND COMBINE
# eurm_lele = sps.load_npz(ROOT_DIR + '/data/lele/ensembled_CLUSTERARTISTScat4-5-6-8-10_online.npz')
# eurm_std = sps.load_npz(ROOT_DIR + '/data/lele/ensembled_SUBCREATIVA_online.npz')
#
# eurm_ens = combine_two_eurms(eurm_lele, eurm_std, cat_first=[4, 5, 6, 8, 10])

# LOAD MATRICES
eurm_ens = sps.load_npz(ROOT_DIR + '/data/ensembled_creativeFIRE_online.npz')
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
sb.submit(rec_list, name='creative_track', track='creative')
