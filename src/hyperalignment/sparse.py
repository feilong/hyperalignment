import numpy as np
from scipy import sparse


def initialize_sparse_matrix(sls, nv=None, dtype=np.float64):
    """Initialize a sparse matrix based on the searchlights.

    Parameters
    ----------
    sls : list
        A list of searchlights. Each entry is an integer array comprising the indices of a searchlight.
    nv : int, optional
        Number of vertices, by default None. The sparse matrix has a shape of (nv, nv).
    dtype : dtype, optional
        The dtype of the generated sparse matrix, by default np.float64

    Returns
    -------
    mat : csc_matrix
        The initialized sparse matrix which allows fast computations for searchlight algorithms. All of its elements are 0's.
    """
    if nv is None:
        nv = np.concatenate(sls).max() + 1
    mat = sparse.lil_matrix((nv, nv), dtype=dtype)
    for sl in sls:
        mat[np.ix_(sl, sl)] = 1.
    mat = mat.tocsc()
    mat.data = np.zeros_like(mat.data, dtype=mat.data.dtype)
    return mat
