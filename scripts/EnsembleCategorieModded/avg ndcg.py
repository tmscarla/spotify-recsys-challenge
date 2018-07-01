import scipy.sparse as sps
from utils.pre_processing import norm_max_row

import sys
from utils.evaluator import Evaluator
from utils.pretty_printer import Pretty_printer
from utils.datareader import Datareader
from utils.post_processing import eurm_to_recommendation_list_submission
from utils.ensembler import ensembler
import numpy as np
from tqdm import tqdm
from utils.post_processing import  eurm_to_recommendation_list
from utils.submitter import Submitter


if __name__ == '__main__':

    w = []
    best = 0
    for i in range(1, 11):
        arg = np.load("cat" + str(i) + "/best.npy")
        print("cat", i,":", arg[-1])
        best += -float(arg[-1])
    print(best/10)