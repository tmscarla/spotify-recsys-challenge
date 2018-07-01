import scipy.sparse as sps
from scripts.scikit_ensemble.scikit_ensamble import Optimizer
from utils.definitions import *
from utils.datareader import Datareader
from utils.definitions import ROOT_DIR


cat = 1
matrix = list()


from utils.definitions import load_obj
name = load_obj("name")
directory = ROOT_DIR + "/scripts/scikit_ensemble/offline/"
matrix_dict = load_obj("matrix_dict", path="")

m = list()
for n in name[cat-1]:
    m.append(sps.load_npz(directory + matrix_dict[n]))
matrix.append(m)



dr = Datareader(verbose=False, mode = "offline", only_load="False")

opt = Optimizer(matrices_array=matrix[0], matrices_names=name[cat-1],
                dr=dr, cat=cat, start=0, end=1)
del matrix
opt.run()