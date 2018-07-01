import sys
from scipy import sparse
import numpy as np
import utils.pre_processing as pre
from utils.definitions import *
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.pre_processing import *
from utils.post_processing import *
from fast_import import *

dr = Datareader(mode='offline', only_load=True, verbose=False)
ev = Evaluator(dr)
urm = dr.get_urm_with_position(1)

urm_std = dr.get_urm()

rec = CF_UB_BM25(urm=urm, datareader=dr, verbose_evaluation=False)
rec.model(alpha=1, beta=0, k=250)
rec.urm = urm_std
rec.fast_recommend()
res = rec.fast_evaluate_eurm()
print(res[1])

