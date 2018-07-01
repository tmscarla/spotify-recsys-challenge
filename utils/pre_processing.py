from scipy import sparse as sp
from numpy import bincount,log, sqrt, ravel, array, squeeze, asarray, ones
from sklearn.preprocessing import quantile_transform
from scipy.special import boxcox1p
import numpy as np
import time
import sklearn.preprocessing as skp
import sklearn.utils.sparsefuncs as skfun
import utils.sparse as ut
from tqdm import tqdm

def __timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print ('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap

def position_matrix_boost(pos_matrix, target_rows=None, mode='linear',p1=None,p2=None,v1=None,v2=None,n_steps=None,verbose=True):
    pos_matrix = sp.csr_matrix(pos_matrix, copy=True, dtype=np.float)
    if target_rows is None: target_rows=np.arange(0,pos_matrix.shape[0],1)
    data = []
    cols = []
    rows = []
    for id_row in tqdm(target_rows, disable=not verbose):
        row = pos_matrix[id_row]
        if mode == 'linear': new_data = linear(row.data,p1,p2,v1,v2)
        elif mode == 'steps': new_data = steps(row.data,n_steps,v1,v2)
        else: print('MODE string error')
        cols.extend(row.indices.tolist())
        rows.extend(np.full(row.indices.shape[0], id_row).tolist())
        data.extend(new_data)
    m1 = sp.csr_matrix((data,(rows,cols)),shape=pos_matrix.shape)
    ut.inplace_set_rows_zero(X=pos_matrix, target_rows=target_rows)
    return pos_matrix+m1


def steps(data,steps=4,v1=None,v2=None):
    if v2 is None: v2=data.shape[0]
    if v1 is None: v1=1
    p2 = data.shape[0]
    step = (p2+1)/steps
    new_data = []
    for pos in data:
        new_data.append((int(pos/step)+1) * v1)
    return new_data


def linear(data,p1=None,p2=None,v1=None,v2=None):
    if p2 is None: p2=data.shape[0]
    if p1 is None: p1=1
    if v2 is None: v2=data.shape[0]
    if v1 is None: v1=1
    new_data = []
    if data.shape[0]==0 or data.shape[0]==1: return data.tolist()
    interpolation = (v2-v1)/(p2-p1)
    for pos in data:
        if(pos<=p1): new_data.append(v1)
        elif(pos>=p2): new_data.append(v2)
        else: new_data.append(v1+interpolation*(pos-p1))
    return new_data


def norm_l1_row(csr_matrix):
    # assert sp.isspmatrix_csr(csr_matrix), "the matrix you are nromalizing is not csr. (it's slower)"
    return skp.normalize(X=csr_matrix, norm='l1',axis=1, copy=True,return_norm=False)


def norm_l2_row(csr_matrix):
    assert csr_matrix.getformat()=='csr'
    # assert sp.isspmatrix_csr(csr_matrix), "the matrix you are nromalizing is not csr. (it's slower)"
    return skp.normalize(X=csr_matrix, norm='l2',axis=1, copy=True, return_norm=False)


def norm_max_row2(matrix):
    tmp = sp.csr_matrix(matrix)
    if min(tmp.data)<0:
        tmp.data = tmp.data+min(tmp.data)
    return norm_max_row(tmp)

def norm_max_row(csr_matrix):
    # assert sp.isspmatrix_csr(csr_matrix), "the matrix you are nromalizing is not csr. (it's slower)"
    return skp.normalize(X=csr_matrix, norm='max',axis=1,copy=True, return_norm=False)

def norm_box_max_row(matrix, lam=0.11):
    tmp = sp.csr_matrix(matrix.copy())
    tmp.data = boxcox1p(tmp.data,lam)
    tmp = skp.normalize(X=tmp, norm='max',axis=1,copy=True, return_norm=False)
    return tmp

def norm_box_l1_row(matrix):
    tmp = sp.csr_matrix(matrix.copy())
    tmp.data = boxcox1p(tmp.data,0.11)
    tmp = skp.normalize(X=tmp, norm='l1',axis=1, copy=True,return_norm=False)
    return tmp

def norm_quantile_uniform(matrix):
    assert sp.isspmatrix_csr(matrix), "the matrix you are nromalizing is not csr "
    return quantile_transform(matrix, axis=1, n_quantiles=1000, output_distribution='uniform',
                              ignore_implicit_zeros=True, subsample=100000,
                              random_state=None, copy=True)

def norm_l1_col(matrix):
    return skp.normalize(X=matrix, norm='l1',axis=0,copy=True, return_norm=False)


def norm_l2_col(matrix):
    return skp.normalize(X=matrix, norm='l2',axis=0,copy=True, return_norm=False)


def norm_max_col(matrix):
    return skp.normalize(X=matrix, norm='max',axis=0,copy=True, return_norm=False)


def bm25_row( X, K1=1.2, B=0.75):
    # Weighs each row of a sparse matrix by OkapiBM25 weighting
    # calculate idf per term (user)
    X = sp.coo_matrix(X)
    N = float(X.shape[0])
    idf = log(N / (1 + bincount(X.col)))

    # calculate length_norm per document (artist)
    row_sums = ravel(X.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X.tocsr()

def bm25_row_no_challange( X, X2, K1=1.2, B=0.75):
    # Weighs each row of a sparse matrix by OkapiBM25 weighting
    # calculate idf per term (user)
    # X2 matrix with no challange set
    X = sp.coo_matrix(X)
    X2 = sp.coo_matrix(X2)
    N = float(X.shape[0])
    idf = log(N / (1 + bincount(X2.col))) #calculated on the mpd

    # calculate length_norm per document (artist)
    row_sums = ravel(X.sum(axis=1))
    row_sums2 = ravel(X2.sum(axis=1))
    average_length = row_sums2.mean() #avg calculated on the mpd
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X.tocsr()


def bm25_col( X, K1=1.2, B=0.75):

    # Weighs each row of a sparse matrix by OkapiBM25 weighting
    # calculate idf per term (user)
    X = sp.coo_matrix(X)
    N = float(X.shape[1])
    idf = log(N / (1 + bincount(X.row)))

    # calculate length_norm per document (artist)
    col_sums = ravel(X.sum(axis=0))
    average_length = col_sums.mean()
    length_norm = (1.0 - B) + B * col_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.col] + X.data) * idf[X.row]

    return X.tocsr()

def bm25_col_inv( X, K1=1.2, B=0.75):
    
    # Weighs each row of a sparse matrix by OkapiBM25 weighting
    # calculate idf per term (user)
    X = sp.coo_matrix(X)
    N = float(X.shape[0])
    idf = log(N / (1 + bincount(X.row)))

    # calculate length_norm per document (artist)
    col_sums = ravel(X.sum(axis=0))
    average_length = col_sums.mean()
    length_norm = (1.0 - B) + B * col_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data / (K1 + 1.0) * (K1 * length_norm[X.col] + X.data) * idf[X.row]

    return X.tocsr()

def bm25_row_inv( X, K1=1.2, B=0.75):
    # Weighs each row of a sparse matrix by OkapiBM25 weighting
    # calculate idf per term (user)
    X = sp.coo_matrix(X)
    N = float(X.shape[0])
    idf = log(N / (1 + bincount(X.col)))

    # calculate length_norm per document (artist)
    row_sums = ravel(X.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data / (K1 + 1.0) * (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X.tocsr()


def base_tfidf_row( X ):

    # TFIDF each row of a sparse amtrix
    X = sp.coo_matrix(X)
    N = float(X.shape[0])
    # calculate IDF
    idf = log(N / (1 + bincount(X.col)))

    print(idf[X.col])
    # apply TF-IDF adjustment
    X.data = sqrt(X.data) * idf[X.col]
    return X.tocsr()


def base_tfidf_col( X ):
    X = X.T
    X = base_tfidf_row(X)
    X = X.T
    return X.tocsr()


def random_sampling_cat(urm, datareader, cat, n_samples):
    """
    Set n_samples equal to zero for the selected category. Samples will be equally spaced.
    :param urm: the user rating matrix
    :param datareader: a Datareader object
    :param cat: the selected category
    :param n_samples: number of tracks to set equal to zero
    :return: urm: a copy of the eurm with zeros eliminated
    """

    urm = urm.copy()
    cat_pids = datareader.get_test_pids(cat=cat)

    for pid in cat_pids:
        row_start = urm.indptr[pid]
        row_end = urm.indptr[pid+1]

        length = len(urm.data[row_start:row_end])
        step = int(length / n_samples)

        indices = [step * x for x in range(n_samples) if x < length]
        urm.data[row_start:row_end][indices] = 0

    urm.eliminate_zeros()

    return urm


def tfidf(sparse_matrix, tf_type='none', idf_type='idf',  verbose=False, k=0.5):
    """
    :param sparse_matrix:
                gimme your icm. (sparse, csr plz)
                row : elements
                columns: features
    :param tf:
                'none' or 'raw' TF IS USELESS (does nothing)
                'binary'        like none, but eliminating the duplicates >> 0/1
                'tf_normal      term frequency:
                            (Number of times the song appears in a playl) / (Total number of songs in the playlist)
                'tf_duplicates'            term frequency //(
                            (Number of times the song appears in a playl) / (Total number of songs in the document)
                'log'           log normalization
                'double_k'      normalization but with parameter K (default 0.5

    :param idf:
                'none' or'' do not apply idf
                'idf'       inverse document frequency
                'smooth'    inverse document frequency smooth
                'max'       inverse document frequency max
                'square'    inverse dovument freq square root
                'squaresmooth'    inverse dovument freq square root smooth
                'prob'      probabilistic inverse document frequency

    :param k: tune double normalization
    :return: your lovely icm, tuned as you said
    """


    icm = sp.csr_matrix(sparse_matrix.copy(), dtype=float)
    icm.eliminate_zeros()
    N = np.count_nonzero( np.diff(icm.indptr)) #num of nonempty playlists
    N2 = icm.shape[0]

    nt = np.diff(icm.indptr)           # number of UNIQUE tracks of the playlists
    nt_dupli = icm.sum(axis=1).A1      # number of tracks in playlists with duplicates
    popularity = np.diff(icm.tocsc().indptr)  # popularity
    npl_dupli = icm.sum(axis=0).A1

    def __apply_tf__(icm):
        if tf_type == 'none' or tf_type =='raw' or tf_type=='':
            pass
        elif tf_type == 'binary':
            icm.data = ones(len(icm.data))
        elif tf_type=='tf_normal':
            skfun.inplace_row_scale(icm, 1/nt_dupli)
        elif tf_type=='tf_duplicates':
            skfun.inplace_row_scale(icm, 1/nt)
        elif tf_type=='tf_elduplicates':
            icm.data = ones(len(icm.data))
            skfun.inplace_row_scale(icm, 1/nt)
        elif tf_type== 'log':
            icm.data += ones(len(icm.data))
            icm.data = log(icm.data)
        elif tf_type =='double_k':
            max_per_playlist = np.maximum.reduceat(icm.data, icm.indptr[:-1])
            max_per_playlist[np.diff(icm.indptr) == 0] = 0
            skfun.inplace_row_scale(icm, k / max_per_playlist)
            icm.data = k + icm.data
        else:
            raise AttributeError("nigga wut? idf ["+tf_type+"] not found")

    def __apply_idf__(icm):
        if idf_type == 'none' or idf_type=='':
            pass
        elif idf_type == 'idf':
            skfun.inplace_column_scale(icm, np.log10(N/popularity))
        elif idf_type == 'idfshrinked':
            skfun.inplace_column_scale(icm, np.log10(N / 1+popularity))
        elif idf_type =='smooth':
            skfun.inplace_column_scale(icm, np.log10(1+(N/popularity) ))
        elif idf_type =='max':
            nt_max = np.max(nt)
            skfun.inplace_column_scale(icm, np.log10( nt_max/(1+popularity) ))
        elif idf_type =='square':
            skfun.inplace_column_scale(icm,  sqrt( N/popularity))
        elif idf_type == 'squaresmooth':
            skfun.inplace_column_scale(icm, sqrt(1+ N/popularity ))
        elif idf_type =='prob':
            skfun.inplace_column_scale(icm,np.log10( (N-popularity)/popularity ))
        else:
            raise AttributeError("nigga wut? idf ["+idf_type+"] not found")

    __apply_tf__(icm=icm)

    __apply_idf__(icm=icm)

    return icm


if __name__ == '__main__':
    pass

    # a = sp.load_npz('../data/test1/matrices')
