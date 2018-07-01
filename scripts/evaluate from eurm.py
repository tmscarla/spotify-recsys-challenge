"""
simple code to evaluate a recommender, you need a sparse matrix of shape (10k,2,2kk)
"""


from utils.datareader import Datareader
from utils.evaluator import Evaluator
import utils.post_processing as post
import scipy.sparse as sps

filename = "file.npz"
output_name = "matrix_factorization"


if __name__ == '__main__':

    dr = Datareader(mode="offline", only_load=True)
    ev = Evaluator(dr)
    pids = dr.get_test_playlists().transpose()[0]
    algorithm_eurm_full = sps.load_npz(filename)
    algorithm_eurm_small= algorithm_eurm_full[pids]

    ev.evaluate(post.eurm_to_recommendation_list(algorithm_eurm_small), name=output_name,
                verbose=True, show_plot=True, save=True)