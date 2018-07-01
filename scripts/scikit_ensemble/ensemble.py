import scipy.sparse as sps
import numpy as np
from utils.ensembler import ensembler
from utils.definitions import  load_obj
from utils.post_processing import  eurm_to_recommendation_list
from utils.post_processing import  eurm_remove_seed
from utils.submitter import Submitter
from utils.evaluator import Evaluator
from utils.datareader import Datareader
from utils.definitions import *

import itertools

def flatten(L):
    return list(set([val for sublist in L for val in sublist]))

def reorder(dict, order):
    assert len(dict)==len(order)
    ret = [dict[k] for k in order]
    return ret

if __name__ == '__main__':

    name = load_obj("name")
    mode = "online"
    type = "unique"

    print("[ Initizalizing Datereader ]")
    dr = Datareader(verbose=False, mode=mode, only_load="False")
    directory = ROOT_DIR+"/scripts/scikit_ensemble/"+mode+"/"
    w = []
    print("[ Loading weights ]")
    for i in range(1, 11):
        arg = load_obj("best/cat" + str(i) + "")
        w.append(reorder(dict(arg[:len(arg) - 1][0]), name[i-1]))




    print("[ Loading matrix name ]")
    if mode == "offline":
        matrix_dict = load_obj("matrix_dict", path="")
        dir = "offline/"

    if mode == "online":
        matrix_dict = load_obj("matrix_dict_online", path="")
        dir = "online/"


    _name = flatten(name)
    loaded_matrix = dict(zip(_name, [eurm_remove_seed(sps.load_npz(directory + matrix_dict[n]), dr) for n in _name]))



    matrix = []

    if type == "unique":
        print("[ Loading cat 1 ]")
        cat = 1
        m = list()
        for n in name[cat-1]:
            m.append(loaded_matrix[n][0:1000])
        matrix.append(m)

        print("[ Loading cat 2 ]")
        cat = 2
        m = list()
        for n in name[cat-1]:
            m.append(loaded_matrix[n][1000:2000])
        matrix.append(m)

        print("[ Loading cat 3 ]")
        cat = 3
        m = list()
        for n in name[cat-1]:
            m.append(loaded_matrix[n][2000:3000])
        matrix.append(m)

        print("[ Loading cat 4 ]")
        cat = 4
        m = list()
        for n in name[cat-1]:
            m.append(loaded_matrix[n][3000:4000])
        matrix.append(m)

        print("[ Loading cat 5 ]")
        cat = 5
        m = list()
        for n in name[cat-1]:
            m.append(loaded_matrix[n][4000:5000])
        matrix.append(m)

        print("[ Loading cat 6 ]")
        cat = 6
        m = list()
        for n in name[cat-1]:
            m.append(loaded_matrix[n][5000:6000])
        matrix.append(m)

        print("[ Loading cat 7 ]")
        cat = 7
        m = list()
        for n in name[cat-1]:
            m.append(loaded_matrix[n][6000:7000])
        matrix.append(m)

        print("[ Loading cat 8 ]")
        cat = 8
        m = list()
        for n in name[cat-1]:
            m.append(loaded_matrix[n][7000:8000])
        matrix.append(m)

        print("[ Loading cat 9 ]")
        cat = 9
        m = list()
        for n in name[cat-1]:
            m.append(loaded_matrix[n][8000:9000])
        matrix.append(m)

        print("[ Loading cat 10 ]")
        cat = 10
        m = list()
        for n in name[cat-1]:
            m.append(eurm_remove_seed(loaded_matrix[n],dr)[9000:10000])
        matrix.append(m)

        rprec = []
        for i in range(0, 10):
            print("[ Ensembling cat", i+1, "]")
            rprec.append(ensembler(matrix[i], w[i], normalization_type="max"))
        res = sps.vstack(rprec).tocsr()

        import time
        n = "ensemble-"+mode+"-data-"+time.strftime("%x")+"-"+time.strftime("%X")
        n = n.replace("/", "_")
        n = n.replace(":", "_")
        sps.save_npz("results/"+n+".npz", res)




        res = eurm_to_recommendation_list(res, datareader=dr)

        if mode == "offline":
            print("[ Initizalizing Evaluator ]")
            ev = Evaluator(dr)

            ev.evaluate(res, name="ens")

        if mode == "online":
            print("[ Initizalizing Submitter ]")
            sb = Submitter(dr)
            sb.submit(recommendation_list=res, name="nuova_submission", track="main", verify=True,
                      gzipped=False)

