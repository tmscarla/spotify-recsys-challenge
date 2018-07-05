import sys
from utils.definitions import ROOT_DIR
from scripts.top_pop_p import Top_pop_p
import scipy.sparse as sps

arg = sys.argv[1:]
mode = arg[0]


t = Top_pop_p()
eurm = t.get_top_pop_album(mode)
sps.save_npz(ROOT_DIR+"/recommenders/script/main/"+mode+"_npz/top_pop_2_album_"+mode+".npz", eurm)
# from utils.datareader import Datareader
# dr = Datareader(verbose=False, mode='offline', only_load=True)
# from utils.evaluator import Evaluator
# from utils.post_processing import eurm_to_recommendation_list
# ev = Evaluator(dr)
# ev.evaluate(recommendation_list=eurm_to_recommendation_list(eurm), name="prova_test")