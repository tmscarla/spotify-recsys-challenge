import sys
from recommenders.script.main.top_pop_p import Top_pop_p
import scipy.sparse as sps
from utils.definitions import ROOT_DIR

arg = sys.argv[1:]
mode = arg[0]


t = Top_pop_p()

eurm = t.get_top_pop_track(mode)
sps.save_npz(ROOT_DIR+"/recommenders/script/main/"+mode+"_npz/top_pop_2_track_"+mode+".npz", eurm)

# from utils.datareader import Datareader
# dr = Datareader(verbose=False, mode='offline', only_load=True)
# from utils.evaluator import Evaluator
# from utils.post_processing import eurm_to_recommendation_list
# ev = Evaluator(dr)
# ev.evaluate(recommendation_list=eurm_to_recommendation_list(eurm), name="prova_test")