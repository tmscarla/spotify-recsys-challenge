"""
EXAMPLES, OFFLINE:
    python ensemble.py   base_simo       offline
    python ensemble.py   experiment2     offline
    python ensemble.py   experiment3     offline

EXAMPLES ONLINE
    python ensemble.py   configuration_name   online  main
    python ensemble.py   experiment1          online  creative
    python ensemble.py   experiment2          online  creative

"""

from utils.post_processing import eurm_to_recommendation_list,eurm_remove_seed
from utils.pre_processing import norm_max_row, norm_l1_row,norm_l2_row, norm_box_l1_row, \
    norm_box_max_row,norm_quantile_uniform, norm_max_row2
from utils.evaluator import Evaluator
from utils.datareader import Datareader
from utils.submitter import Submitter
from utils.definitions import ROOT_DIR, dump_params_dict,read_params_dict
import scipy.sparse as sps
from tqdm import tqdm
import sys,os,json




if __name__ == '__main__':

    configuration_name = sys.argv[1]
    mode = sys.argv[2]

    assert mode == 'online' or mode == 'offline'

    ### types of nromalizations
    norms=dict()
    norms['max'] = norm_max_row
    norms['l1']  = norm_l1_row
    norms['l2']  = norm_l2_row
    norms['box_max'] = norm_box_max_row
    norms['box_l1']  = norm_box_l1_row
    norms['q_uni']   = norm_quantile_uniform
    norms['max2']   = norm_max_row2

    dr = Datareader(verbose=False, mode=mode, only_load=True)

    matrices_names = read_params_dict(ROOT_DIR+'/bayesian_scikit/'+configuration_name+'/name_settings')
    file_locations = read_params_dict(ROOT_DIR + '/bayesian_scikit/bayesian_common_files/file_locations_' + mode)


    # LOAD MATRICES
    matrices_loaded=dict()
    all_matrices_names = set()
    for cat in range(1,11):

        with open(ROOT_DIR+'/bayesian_scikit/'+configuration_name + '/best_params/cat'+str(cat)+'_params_dict') as f:
            best_params_dict = json.load(f)

        for name, value_from_bayesian in best_params_dict.items():
            all_matrices_names.add(name)
    for name in  tqdm(all_matrices_names,desc='loading matrices'):
        if name not in matrices_loaded.keys() and name!='norm':
            matrices_loaded[name] = eurm_remove_seed(sps.load_npz(file_locations[name]), dr)

    rec_list = [[] for x in range(10000)]
    eurms_cutted = [[] for x in range(10)]

    # BUILDING THE EURM FROM THE PARAMS
    for cat in tqdm(range(1,11),desc="summing up the matrices"):

        start_index = (cat - 1) * 1000
        end_index = cat * 1000

        best_params_dict = read_params_dict(name='cat' + str(cat) + '_params_dict',
                 path=ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/best_params/')


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


    sps.save_npz(file='../'+configuration_name+'/ensembled_'+configuration_name+'_'+mode, matrix=eurm)

    if mode=='offline':
        ev = Evaluator(dr)
        ev.evaluate(recommendation_list=rec_list, name=configuration_name)
    else:
        sb = Submitter(dr)
        sb.submit(recommendation_list=rec_list, name=configuration_name)


