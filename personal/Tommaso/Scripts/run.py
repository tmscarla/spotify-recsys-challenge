import sys
from scipy import sparse
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


def offline():
    # INIT
    dr = Datareader(mode='offline', only_load=True, verbose=False)
    ev = Evaluator(dr)

    # LOAD AND COMBINE
    eurm_lele = sparse.load_npz(ROOT_DIR + '/data/lele/ensembled_CLUSTERARTISTScat4-5-6-8-10_offline.npz')
    eurm_std = sparse.load_npz(ROOT_DIR + '/data/lele/ensembled_SUBCREATIVA_offline.npz')

    eurm_ens = combine_two_eurms(eurm_lele, eurm_std, cat_first=[4, 5, 6, 8, 10])

    # LOAD
    # eurm_ens = sparse.load_npz(ROOT_DIR + '/data/ensembled_creativeFIRE_offline.npz')
    sim = sparse.load_npz(ROOT_DIR + '/data/sim_offline.npz')

    # TOPBOOST
    # topb = TopBoost(dr, eurm_ens, sim)
    # eurm_ens = topb.boost_eurm(categories=[9], top_k=100, gamma=0.01)

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

    # MATCHBOOST
    # mb = MatchBoost(datareader=dr, eurm=eurm_ens, top_k_alb=5000, top_k_art=10000)
    # eurm_ens, pids = mb.boost_eurm(categories='all', k_art=300, k_alb=300, gamma_art=0.1, gamma_alb=0.1)

    # EVALUATION
    rec_list = eurm_to_recommendation_list(eurm_ens, datareader=dr)
    sparse.save_npz('FINAL.npz', eurm_ens)
    ev.evaluate(rec_list, name='LELE_boosts.csv')


def online():
    # INIT
    dr = Datareader(mode='online', only_load=True, verbose=False)
    sb = Submitter(dr)

    # LOAD AND COMBINE
    eurm_lele = sparse.load_npz(ROOT_DIR + '/data/jess/ensembled_CLUSTERARTISTS_CREATIVA_cat3-4-5-8-10_online.npz')
    eurm_std = sparse.load_npz(ROOT_DIR + '/data/jess/ensembled_creativeFIRE_online.npz')

    eurm_ens = combine_two_eurms(eurm_lele, eurm_std, cat_first=[3, 4, 5, 8, 10])

    # LOAD MATRICES
    # eurm_ens = sparse.load_npz(ROOT_DIR + '/data/ensembled_creativeFIRE_online.npz')
    sim = sparse.load_npz(ROOT_DIR + '/data/sim_online.npz')

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

    # MATCHBOOST
    # mb = MatchBoost(datareader=dr, eurm=eurm_ens, top_k_alb=5000, top_k_art=10000)
    # eurm_ens, pids = mb.boost_eurm(categories='all', k_art=20, k_alb=20, gamma_art=1.0, gamma_alb=1.0)

    # SUBMISSION
    rec_list = eurm_to_recommendation_list_submission(eurm_ens, datareader=dr)
    sb.submit(rec_list, name='ens_30_june_jess+lele_boosts', track='creative')


def grid_holeboost():
    datareader = Datareader(mode='offline', only_load=True, verbose=False)
    ev = Evaluator(datareader)

    # LOAD AND COMBINE
    eurm_lele = sparse.load_npz(ROOT_DIR + '/data/lele/ensembled_CLUSTERARTISTScat4-5-6-8-10_offline.npz')
    eurm_std = sparse.load_npz(ROOT_DIR + '/data/lele/ensembled_SUBCREATIVA_offline.npz')

    eurm_ens = combine_two_eurms(eurm_lele, eurm_std, cat_first=[4, 5, 6, 8, 10])
    sim_offline = sparse.load_npz(ROOT_DIR + '/data/sim_offline.npz')

    for k in [50, 100, 150, 200, 250, 300, 350, 400]:
        for gamma in [1, 2, 5, 10]:

            h = HoleBoost(similarity=sim_offline, eurm=eurm_ens, datareader=datareader, norm=norm_l1_row)
            eurm_ens_boosted = h.boost_eurm(categories=[8, 10], k=k, gamma=gamma)
            rec_list = eurm_to_recommendation_list(eurm_ens_boosted, datareader=datareader)

            print('--------------------------------------------------------------------------')
            print('K =', k)
            print('G =', gamma)
            ev.evaluate(rec_list, name='hb', save=False)

def grid_tailboost():
    datareader = Datareader(mode='offline', only_load=True, verbose=False)
    ev = Evaluator(datareader)

    # LOAD AND COMBINE
    eurm_lele = sparse.load_npz(ROOT_DIR + '/data/lele/ensembled_CLUSTERARTISTScat4-5-6-8-10_offline.npz')
    eurm_std = sparse.load_npz(ROOT_DIR + '/data/lele/ensembled_SUBCREATIVA_offline.npz')

    eurm_ens = combine_two_eurms(eurm_lele, eurm_std, cat_first=[4, 5, 6, 8, 10])
    sim = sparse.load_npz(ROOT_DIR + '/data/sim_offline.npz')

    # TAILBOOST
    for lt in [2, 3, 5, 6, 10]:
        for k in [20, 50, 80, 100, 150]:
            for g in [0.005, 0.01, 0.02, 0.05]:

                tb = TailBoost(similarity=sim, eurm=eurm_ens, datareader=datareader, norm=norm_l2_row)
                eurm_ens = tb.boost_eurm(categories=[9, 7, 6, 5],
                                         last_tracks=[lt, lt, lt, lt],
                                         k=[k, k, k, k],
                                         gamma=[g, g, g, g])
                rec_list = eurm_to_recommendation_list(eurm_ens, datareader=datareader)

                print('--------------------------------------------------------------------------')
                print('LT =', lt)
                print('K =', k)
                print('G =', g)
                ev.evaluate(rec_list, name='tb', save=False)


if __name__ == '__main__':
    if sys.argv[1] == '-offline':
        offline()
    elif sys.argv[1] == '-online':
        online()
    elif sys.argv[1] == '-hb':
        grid_holeboost()
    elif sys.argv[1] == '-tb':
        grid_tailboost()

