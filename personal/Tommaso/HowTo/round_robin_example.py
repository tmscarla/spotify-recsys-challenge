from scipy import sparse as sps
from tqdm import tqdm

from utils.datareader import Datareader
from utils.definitions import ROOT_DIR
from utils.evaluator import Evaluator
from utils.post_processing import eurm_to_recommendation_list
from utils.round_robin import RoundRobin

# Datareader
dr = Datareader(mode='offline', only_load=True)
ev = Evaluator(dr)

# Load matrices
print('Loading matrices...')

eurm_knn_album = sps.load_npz(ROOT_DIR + "/data/offline/ENSEMBLE - KNN CONTENT ITEM - album - top_k = 100 - sm_type = cosine - shrink = 100.npz")
eurm_knn_artist = sps.load_npz(ROOT_DIR + "/data/offline/ENSEMBLE - KNN CONTENT ITEM - artist - top_k = 100 - sm_type = cosine - shrink = 100.npz")
eurm_rp3 = sps.load_npz(ROOT_DIR + "/data/offline/ENSEMBLE - RP3BETA - top_k=100 - shrink=100 - alpha=0.5 - beta=0.4.npz")
eurm_nlp = sps.load_npz(ROOT_DIR + "/data/eurm_nlp_offline.npz")

# Convert in rec_list
rec_list_rp3 = eurm_to_recommendation_list(eurm_rp3)
rec_list_knn_album = eurm_to_recommendation_list(eurm_knn_album)
rec_list_knn_artist = eurm_to_recommendation_list(eurm_knn_artist)
rec_list_nlp = eurm_to_recommendation_list(eurm_nlp)

# Round Robin
RR = RoundRobin([rec_list_rp3, rec_list_knn_album, rec_list_knn_artist, rec_list_nlp], weights=None)
rec_list_rr = rec_list_rp3

for k in [20, 50, 70, 90, 100, 120, 150, 180, 200, 230]:
    for i in tqdm(range(1000, len(rec_list_rp3)), desc='Round Robin ' + k):
        prediction = RR.rr_avg(playlist_index=i, rec_index=0, cut_off=k, K=k)
        #prediction = RR.rr_jmp(playlist_index=i, K=k)

        for t in range(len(prediction)):
            rec_list_rr[i][t] = prediction[t]

    print(ev.evaluate(rec_list_rr, name='rr_' + str(k), show_plot=False, save=False, return_overall_mean=True, verbose=False))
