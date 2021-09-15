import numpy as np
from scipy import sparse


def initialize_sparse_matrix(sls, nv=None, dtype=np.float64):
    if nv is None:
        nv = np.concatenate(sls).max() + 1
    mat = sparse.lil_matrix((nv, nv), dtype=dtype)
    for sl in sls:
        mat[np.ix_(sl, sl)] = 1.
    mat = mat.tocsc()
    mat.data = np.zeros_like(mat.data, dtype=mat.data.dtype)
    return mat
