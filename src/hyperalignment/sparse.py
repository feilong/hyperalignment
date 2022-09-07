import os
import numpy as np
from scipy import sparse


def initialize_sparse_matrix(sls, nv=None, dtype=np.float64, cache_fn=None):
    """Initialize a sparse matrix based on the searchlights.

    Parameters
    ----------
    sls : list
        A list of searchlights. Each entry is an integer array comprising the indices of a searchlight.
    nv : int, optional
        Number of vertices, by default None. The sparse matrix has a shape of (nv, nv).
    dtype : dtype, optional
        The dtype of the generated sparse matrix, by default np.float64
    cache_fn : {None, str}
        The file name of the cached sparse matrix. If the file exists, it will
        be loaded to avoid repeated computations. If it is `None`, no cached
        file will be used and the matrix will be computed.

    Returns
    -------
    mat : csc_matrix
        The initialized sparse matrix which allows fast computations for searchlight algorithms. All of its elements are 0's.
    """
    if cache_fn is not None and os.path.exists(cache_fn):
        return sparse.load_npz(cache_fn)
    if nv is None:
        nv = np.concatenate(sls).max() + 1
    mat = sparse.lil_matrix((nv, nv), dtype=dtype)
    for sl in sls:
        mat[np.ix_(sl, sl)] = 1.
    mat = mat.tocsc()
    mat.data = np.zeros_like(mat.data, dtype=mat.data.dtype)
    if cache_fn is not None:
        os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
        sparse.save_npz(cache_fn, mat)
    return mat
