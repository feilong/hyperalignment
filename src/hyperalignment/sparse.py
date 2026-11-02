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
    if len(sls):
        indices = np.concatenate(sls).astype(np.int64)
    else:
        indices = np.zeros(0, dtype=np.int64)
    if nv is None:
        if not len(indices):
            raise ValueError("`nv` must be specified when `sls` is empty.")
        nv = int(indices.max()) + 1
    # The CSR constructor below does not check bounds, so validate them here.
    if len(indices) and (indices.min() < 0 or indices.max() >= nv):
        raise IndexError(f"Searchlight indices must be in [0, {nv}) for nv={nv}.")

    # Membership matrix: row i is the indicator of searchlight i. The product
    # B.T @ B is nonzero exactly where two vertices share a searchlight, which
    # is the sparsity pattern of a searchlight model.
    indptr = np.zeros(len(sls) + 1, dtype=np.int64)
    np.cumsum([len(sl) for sl in sls], out=indptr[1:])
    data = np.ones(len(indices), dtype=np.float32)
    B = sparse.csr_matrix((data, indices, indptr), shape=(len(sls), nv))

    mat = (B.T @ B).tocsc()
    mat.sum_duplicates()
    mat.data = np.zeros(mat.nnz, dtype=dtype)
    if cache_fn is not None:
        os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
        sparse.save_npz(cache_fn, mat)
    return mat
