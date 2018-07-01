"""

EXAMPLE FOR CAT 9, maximizing ndcg with norm_max_row, :

python generic_bayesian.py 8 ndcg l1 quni

"""

import sys
import time
import json, os

from skopt import gp_minimize
from skopt.space import Real
from utils.post_processing import eurm_to_recommendation_list,eurm_remove_seed
from utils.pre_processing import norm_max_row,norm_max_row2, norm_l1_row,norm_l2_row, norm_box_max_row, norm_box_l1_row,norm_quantile_uniform
from utils.evaluator import Evaluator
from utils.datareader import Datareader
from utils.definitions import read_params_dict,dump_params_dict,ROOT_DIR,save_obj, load_obj

import numpy as np
import scipy.sparse as sps


def print_cat():
    print()
    for i in range(1, 11):
        if i == cat:
            print('| CAT' + str(cat), end='\t')
        else:
            print('|', end='\t')
    print('\n')


def pretty_print(ris, x, start_time, finished=False):
    elapsed = int(time.time() - start_time)
    sys.stdout.write("\033[1;31m")
    print(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -', end='')
    if finished:
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   FINISHED  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print_cat()
    print("NAME:",configuration_name)
    sys.stdout.write("\033[0;0m")
    print('run:' + str(global_counter) + ', this_run_time: ' + str(elapsed) + 's,' + str(
        target_metric).upper() + 'BEST:%.4f' % (-ris), '\nVALUES:')

    for i in range(len(x)):
        print(matrices_names[i], "%.2f" % (x[i]), end="\t")
    print('\n')


def objective_function(x):
    global best_score, global_counter, best_params, start_time, x0,y0

    eurm = sum(x[i] * matrix for i, matrix in enumerate(matrices_array))

    # real objective function
    ris = -ev.evaluate_single_metric(eurm_to_recommendation_list(eurm, cat=cat, remove_seed=False, verbose=False),
                                     verbose=False,
                                     cat=cat,
                                     name="ens" + str(cat),
                                     metric=target_metric,
                                     level='track')

    if x0 is None:
        x0 = [x]
        y0 = [ris]
    else:
        x0.append(x)
        y0.append(ris)




    global_counter += 1
    if ris < best_score:
        best_score = ris
        best_params = x.copy()

        pretty_print(ris, x, start_time)

        best_params_dict = dict(zip(matrices_names+['norm'], x.copy()+[norm_name]))
        if not os.path.exists(ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/best_params/'):
            os.mkdir(ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/best_params/')
        dump_params_dict(your_dict=best_params_dict, name= "cat" + str(cat) + "_params_dict",
                 path=ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/best_params/')
    elif global_counter%25==0:
        pretty_print(ris, x, start_time)

    ## print and save memory x0,y0 every 25 calls
    if global_counter%25==0:
        if not os.path.exists(ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/memory/'):
            os.mkdir(ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/memory/')

        save_obj(x0, "cat" + str(cat) + "_x0_MEMORY",
                 path=ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/memory/')

        save_obj(y0, "cat" + str(cat) + "_y0_MEMORY",
                 path=ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/memory/')


    #### condition with no results
    if global_counter == 10 and best_score ==0:
        best_params_dict = dict(zip(matrices_names + ['norm'], [0.0 for a in range(len(x))] + [norm_name]))
        if not os.path.exists(ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/best_params/'):
            os.mkdir(ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/best_params/')
        dump_params_dict(your_dict=best_params_dict, name="cat" + str(cat) + "_params_dict",
                         path=ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/best_params/')
        pretty_print(ris, x, start_time)
        print("CAT"+str(cat)+" HAS NO lines, EXIT ")
        print("CAT"+str(cat)+" HAS NO lines, EXIT ")
        print("CAT"+str(cat)+" HAS NO lines, EXIT ")
        print("CAT"+str(cat)+" HAS NO lines, EXIT ")
        print("CAT"+str(cat)+" HAS NO lines, EXIT ")
        print("CAT"+str(cat)+" HAS NO lines, EXIT ")
        print("CAT"+str(cat)+" HAS NO lines, EXIT ")
        print("CAT"+str(cat)+" HAS NO lines, EXIT ")
        print("CAT"+str(cat)+" HAS NO lines, EXIT ")
        print("CAT"+str(cat)+" HAS NO lines, EXIT ")
        exit()

    start_time = time.time()
    return ris


if __name__ == '__main__':
    dr = Datareader(verbose=False, mode = "offline", only_load=True)
    ev = Evaluator(dr)

    cat = int(sys.argv[1])
    target_metric = sys.argv[2]
    norm_name = sys.argv[3]
    configuration_name = sys.argv[4]


    if norm_name =='max':
        norm = norm_max_row
    elif norm_name =='l1':
        norm = norm_l1_row
    elif norm_name =='l2':
        norm = norm_l2_row
    elif norm_name == 'box_max':
        norm = norm_box_max_row
    elif norm_name == 'box_l1':
        norm = norm_box_l1_row
    elif norm_name == 'q_uni':
        norm = norm_quantile_uniform
    elif norm_name == 'max2':
        norm = norm_max_row2
    else:
        raise ValueError( "norm not found, what is "+norm_name+" ?")


    best_score = 0
    best_params = []
    verbose = True
    calls_constant = 60

    start_index = (cat-1)*1000
    end_index = cat*1000
    global_counter=0

    x0 = None
    y0 = None

    if os.path.isfile(ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/memory/cat'+ str(cat)+'_y0_MEMORY.pkl') and \
            os.path.isfile(ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/memory/cat' + str(cat) + '_x0_MEMORY.pkl'):
        x0 = load_obj('cat' + str(cat) + '_x0_MEMORY', path= ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/memory/')
        y0 = load_obj('cat' + str(cat) + '_y0_MEMORY', path= ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/memory/')
        global_counter = len(y0)
        print("[ CAT"+str(cat)+" : RESUMING FROM RUN", global_counter, "]")

    print("[ CAT "+str(cat)+": STARTING, NOW LOADING MATRICES ]")
    matrices_names = read_params_dict(ROOT_DIR+'/bayesian_scikit/'+configuration_name+'/name_settings')[cat-1]
    file_locations = read_params_dict(ROOT_DIR+'/bayesian_scikit/bayesian_common_files/file_locations_offline')

    matrices_array = [norm( eurm_remove_seed( sps.load_npz(file_locations[x]), dr)[start_index:end_index]) for x in matrices_names ]

    del dr
    start_time=time.time()

    space  = [Real(0, 100, name=x) for x in matrices_names]
    res = gp_minimize(objective_function,  space,
                base_estimator=None,
                n_calls=450+len(matrices_array)*calls_constant, n_random_starts=100,
                acq_func='gp_hedge',
                acq_optimizer='auto',
                x0=x0, y0=y0,
                random_state=None, verbose=False,
                callback=None, n_points=100,
                n_restarts_optimizer=10,
                xi=0.012, kappa=1.96,
                noise='gaussian', n_jobs=3)

    # if not os.path.exists(ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/results_for_plotting/'):
    #     os.mkdir(ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/results_for_plotting/')
    # save_obj(res, "cat" + str(cat) + "_RES",
    #          path=ROOT_DIR + '/bayesian_scikit/' + configuration_name + '/results_for_plotting/')


    pretty_print(best_score,best_params,start_time,finished=True)
