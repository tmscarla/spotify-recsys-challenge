import scipy.sparse as sp
import numpy as np
import mkl

def estimate_memory_matrix_product(matrix, k):
    row = matrix.shape[0]
    d = row * k * 32 / 8 / 1024 / 1024
    maxk = 2147483647/(row)
    dmax = row * maxk * 32 / 8 / 1024 / 1024
    print('array data (float32) = %d MB, array row (int32) = %d MB, array col (int32)= %d MB'%(d,d,d))
    print('total = %d MB \t (%.2f GB)'%(3*d,3*d/1024))
    
def max_k_matrix_product(matrix):
    row = matrix.shape[0]
    maxk = 2147483647/(row)
    dmax = row * maxk * 32 / 8 / 1024 / 1024
    print('max value K due to integer range limit = %d (size in memory %.2f GB)'%(maxk,3*dmax/1024))

def get_mkl_threads():
    print('mkl max threads: %d'%mkl.get_max_threads())

def set_mkl_threads(num_threads=1):
    try:
        mkl.set_num_threads(num_threads)
        assert(mkl.get_max_threads()==num_threads)
    except:
        print('!!! ERROR: setting mkl num threads !!!')

def show_config_numpy(num_threads):
    np.__config__.show()
