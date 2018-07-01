import scipy.sparse as sp
import numpy as np
from recommenders.similarity.tversky import tversky_similarity as tversky
from recommenders.similarity.dot_product import dot_product as dot
from recommenders.similarity.cosine import cosine_similarity as cosine
from recommenders.similarity.jaccard import jaccard_similarity as jaccard, tanimoto_similarity as tanimoto
from recommenders.similarity.p3alpha_rp3beta import p3alpha_rp3beta_similarity as rp3beta
from recommenders.similarity.p3alpha_rp3beta import p3alpha_similarity as p3alpha
from recommenders.similarity.dice import dice_similarity as dice
from sklearn.preprocessing import normalize

COSINE = "cosine"
JACCARD = "jaccard"
AS_COSINE = "as_cosine"
P_3_AlPHA = "p3aplha"
TVERSKY = "tversky"
R_P_3_BETA = "rp3beta"
TANIMOTO = "tanimoto"
DICE = "dice"
