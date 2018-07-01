import sys

from skopt import gp_minimize
from skopt.space import Real, Integer
from utils.definitions import load_obj, save_obj
from utils.post_processing import eurm_to_recommendation_list,eurm_remove_seed, shift_rec_list_cutoff
from utils.pre_processing import norm_max_row, norm_l1_row
from utils.evaluator import Evaluator
from utils.post_processing import eurm_remove_seed
from utils.datareader import Datareader
from utils.ensembler import ensembler
from utils.definitions import *
import multiprocessing
import scipy.sparse as sps
import numpy as np
import os.path


# Settings
class Optimizer(object):

    def __init__(self, matrices_names, matrices_array, dr, cat, start, end, n_calls=1000, n_random_starts=0.1, n_points=50, step=0.001, verbose=True):
        self.target_metric = 'ndcg'
        self.best_score = 0
        self.best_params = 0
        self.norm = norm_max_row
        self.verbose = verbose


        self.n_cpu = int(multiprocessing.cpu_count()/10)
        if self.n_cpu == 0:
            self.n_cpu = 1
        # Do not edit
        self.start = start
        self.end = end
        self.cat = cat
        self.global_counter = 0
        self.start_index = (cat - 1) * 1000
        self.end_index = cat * 1000
        self.matrices_array = list()
        self.matrices_names = matrices_names
        self.n_calls = n_calls
        self.global_counter = 0
        self.x0 = None
        self.y0 = None
        self.n_random_starts = int(n_calls*n_random_starts)
        self.n_points = n_points
        self.step = step
        # memory_on_disk= False
        self.memory_on_notebook=True
        self.dr = dr
        self.ev = Evaluator(self.dr)

        for matrix in matrices_array:
            self.matrices_array.append(self.norm(eurm_remove_seed(matrix ,datareader=dr)[self.start_index:self.end_index]))

        del self.dr, matrices_array
    def run(self):
        self.x0 = None
        self.y0 = None
        space = [Real(self.start, self.end, name=x) for x in self.matrices_names]
        self.res = gp_minimize(self.obiettivo, space,
                          base_estimator=None,
                          n_calls=self.n_calls, n_random_starts=self.n_random_starts,
                          acq_func='gp_hedge',
                          acq_optimizer='auto',
                          x0=self.x0, y0=self.y0,
                          random_state=None, verbose=self.verbose,
                          callback=None, n_points=self.n_points,
                          n_restarts_optimizer=10,
                          xi=self.step, kappa=1.96,
                          noise='gaussian', n_jobs=self.n_cpu)


    def obiettivo(self, x):

        eurm = sum(x[i] * matrix for i, matrix in enumerate(self.matrices_array))

        # real objective function
        ris = -self.ev.evaluate_single_metric(eurm_to_recommendation_list(eurm, cat=self.cat, remove_seed=False, verbose=False),
                                         verbose=False,
                                         cat=self.cat,
                                         name="ens" + str(self.cat),
                                         metric=self.target_metric,
                                         level='track')
        # memory variables
        if self.x0 is None:
            self.x0 = [[x]]
            self.y0 = [ris]
        else:
            self.x0.append(x)
            self.y0.append(ris)

        self.global_counter += 1
        if ris < self.best_score:
            print("[NEW BEST]")
            self.pretty_print(ris, x)
            self.best_score = ris
            self.best_params = x.copy()
            self.best_params_dict = dict(zip(self.matrices_names, x.copy()))
            b = list()
            if os.path.isfile("best/cat"+str(self.cat)+".plk"):
                b.append(self.best_params_dict)
                b.append(ris)
                save_obj(b, "best/cat"+str(self.cat))
            else:
                b.append(self.best_params_dict)
                b.append(ris)
                save_obj(b, "best/cat"+str(self.cat))
        elif self.verbose:
            self.pretty_print(ris, x)


        return ris


    def pretty_print(self, ris, x):
        print("CAT:", self.cat, "ITER:", self.global_counter, "RES:", ris, end="\tvals:\t")
        for i in range(len(x)):
            print(self.matrices_names[i], "%.4f" % (x[i]), end="\t")
        print()
        print("-"*80)
        print()

