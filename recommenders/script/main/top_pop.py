from utils.datareader import Datareader
import scipy.sparse as sps
import sys
from utils.definitions import ROOT_DIR

arg = sys.argv[1:]
mode = arg[0]

dr = Datareader(verbose=False, mode='offline', only_load=True)
top_pop = dr.get_eurm_top_pop(top_pop_k=750, remove_duplicates=True, binary=True)
sps.save_npz(ROOT_DIR+"/recommenders/script/main/"+mode+"_npz/top_pop.npz", top_pop)
