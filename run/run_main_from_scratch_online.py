
from utils.evaluator import Evaluator
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


if __name__ == '__main__':

    ### types of nromalizations
    norms=dict()
    norms['max'] = norm_max_row
    norms['l1']  = norm_l1_row

    dr = Datareader(verbose=False, mode='online', only_load=True)

    ar1_location = ROOT_DIR+'/recommenders/script/main/online_npz/npz_ar1/'
    ar2_location = ROOT_DIR+'/recommenders/script/main/online_npz/npz_ar2/'
    ar3_location = ROOT_DIR+'/recommenders/script/main/online_npz/npz_ar3/'
    ar4_location = ROOT_DIR+'/recommenders/script/main/online_npz/npz_ar4/'
    main_location= ROOT_DIR+'/recommenders/script/main/online_npz/'

    clusters = [ ('ar1', ar1_location),
                 ('ar2', ar2_location),
                 ('ar3', ar3_location),
                 ('ar4', ar4_location),
                 ('SUBMAIN',main_location)]

    file_names = {            'cb_ar':      "cb_ar_online.npz",
                              'cb_al':      "cb_al_online.npz",
                              'cb_al_ar':   "cb_al_ar_online.npz",
                              'cf_ib':      "cf_ib_online.npz",
                              'cf_ub':      "cf_ub_online.npz",

                              'cf_tom_album': "cf_al_online.npz",
                              'cf_tom_artist': "cf_ar_online.npz",

                              'cf_ar':  "cf_ar_online.npz",
                              'cf_al':  "cf_al_online.npz",

                              'cf_ib_new':  "cf_ib_online.npz",
                              'cf_ub_new':  "cf_ub_online.npz",

                              'hyb_j_main_cat10':  'cb_layer_cat10_online.npz',
                              'hyb_j_main_cat8':  'cb_layer_cat8_online.npz',

                              # from keplero
                              'top_pop': 'top_pop.npz',

                              'nlp_fusion': 'nlp_fusion_online.npz',

                              ##### things for single categories:

                              'top_pop_album_cat2': 'top_pop_2_album_online.npz',
                              'top_pop_track_cat2': 'top_pop_2_track_online.npz',

                              'cb_ib_cat9': 'cb_ib_cat9_online.npz'
                              }

    eurm_list = []

    for cluster in clusters:
        # LOAD MATRICES
        matrices_loaded=dict()
        all_matrices_names = set()
        for cat in range(1,11):
            # print(ROOT_DIR+'/bayesian_scikit/'+cluster[0] + '/best_params/cat'+str(cat)+'_params_dict')
            with open(ROOT_DIR+'/bayesian_scikit/'+cluster[0] + '/best_params/cat'+str(cat)+'_params_dict') as f:
                best_params_dict = json.load(f)

            for name, value_from_bayesian in best_params_dict.items():
                all_matrices_names.add(name)
        for name in  tqdm(all_matrices_names,desc='loading matrices'):
            if name not in matrices_loaded.keys() and name!='norm':
                # print(cluster[1]+name+'_online.npz')
                matrices_loaded[name] = eurm_remove_seed(sps.load_npz(cluster[1]+file_names[name]), dr)

        rec_list = [[] for x in range(10000)]
        eurms_cutted = [[] for x in range(10)]


        # BUILDING THE EURM FROM THE PARAMS
        for cat in tqdm(range(1,11),desc="summing up the matrices"):
            start_index = (cat - 1) * 1000
            end_index = cat * 1000

            best_params_dict = read_params_dict(name='cat' + str(cat) + '_params_dict',
                     path=ROOT_DIR + '/bayesian_scikit/' + cluster[0] + '/best_params/')


            norm = best_params_dict['norm']
            del best_params_dict['norm']
            # cutting and  dot the value from ensemble
            eurms_full = [ value_from_bayesian * norms[norm](matrices_loaded[name][start_index:end_index])
                            for name, value_from_bayesian in best_params_dict.items()]
            # and summing up
            eurms_cutted[cat-1] = sum( [ matrix for matrix in eurms_full] )

            # adding to reclist
            rec_list[start_index:end_index] = eurm_to_recommendation_list(eurm=eurms_cutted[cat-1],
                                                                          cat=cat,
                                                                          verbose=False)[start_index:end_index]

        eurm = eurms_cutted[0]
        for i in range(1,10):
            eurm = sps.vstack([eurm, eurms_cutted[i]])

        eurm_list.append(eurm)

    CLUSTERED_MATRIX = eurm_list[0]+eurm_list[1]+eurm_list[2]+eurm_list[3]

    ev = Evaluator(dr)
    ev.evaluate(recommendation_list=eurm_to_recommendation_list(CLUSTERED_MATRIX), name='clustered_online')

    ENSEMBLED = eurm_list[4]

    ev.evaluate(recommendation_list=eurm_to_recommendation_list(ENSEMBLED), name='ensembled_online')

    ####### POSTPROCESSING #################################################################

    # COMBINE
    FINAL = combine_two_eurms(CLUSTERED_MATRIX, ENSEMBLED, cat_first=[3, 4, 5, 6, 8, 10])
    sim = generate_similarity('online')

    # HOLEBOOST
    hb = HoleBoost(similarity=sim, eurm=FINAL, datareader=dr, norm=norm_l1_row)
    FINAL = hb.boost_eurm(categories=[8], k=300, gamma=1)
    hb = HoleBoost(similarity=sim, eurm=FINAL, datareader=dr, norm=norm_l1_row)
    FINAL = hb.boost_eurm(categories=[10], k=150, gamma=1)

    # TAILBOOST
    tb = TailBoost(similarity=sim, eurm=FINAL, datareader=dr, norm=norm_l2_row)
    FINAL = tb.boost_eurm(categories=[9, 7, 6, 5],
                             last_tracks=[10, 3, 3, 3],
                             k=[100, 80, 100, 100],
                             gamma=[0.01, 0.01, 0.01, 0.01])

    # ALBUMBOOST
    ab = AlbumBoost(dr, FINAL)
    FINAL = ab.boost_eurm(categories=[3, 4, 7, 9], gamma=2, top_k=[3, 3, 10, 40])

    ev.evaluate(recommendation_list=eurm_to_recommendation_list(FINAL), name='main_track_online')



