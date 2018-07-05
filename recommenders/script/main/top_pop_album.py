import sys
from scripts.top_pop_p import Top_pop_p
import scipy.sparse as sps

arg = sys.argv[1:]
mode = arg[0]


if mode == "online":
    t = Top_pop_p()
    eurm = t.get_top_pop_album()
    sps.save_npz("top_pop_2_album_"+mode+".npz", eurm)
