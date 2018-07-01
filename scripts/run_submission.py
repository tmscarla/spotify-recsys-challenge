from scipy import sparse
from boosts.hole_boost import HoleBoost
from boosts.tail_boost import TailBoost
from boosts.album_boost import AlbumBoost
from boosts.match_boost import MatchBoost
from utils.post_processing import *
from utils.submitter import Submitter
from utils.pre_processing import *


def submission(boost, eurm_ens, sim, name):
    """
    Function to create a submission from a eurm with or without boosts.
    :param boost: apply boosts
    :param eurm_ens: eurm from ensemble (10k x 2.2M)
    :param sim: similarity matrix (tracks x tracks)
    :param name: name of the submission
    """

    # INIT
    dr = Datareader(mode='online', only_load=True, verbose=False)
    sb = Submitter(dr)

    if boost:
        # HOLEBOOST
        hb = HoleBoost(similarity=sim, eurm=eurm_ens, datareader=dr, norm=norm_l1_row)
        eurm_ens = hb.boost_eurm(categories=[8, 10], k=300, gamma=5)

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
    sb.submit(rec_list, name=name)


if __name__ == '__main__':

    # SETTINGS
    boost = True
    eurm = sparse.load_npz(ROOT_DIR + '')
    similarity = sparse.load_npz(ROOT_DIR + '')

    submission(boost=boost, eurm_ens=eurm, sim=similarity)
