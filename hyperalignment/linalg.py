import numpy as np
from scipy.linalg import svd, LinAlgError


def safe_svd(X, demean=True):
    if demean:
        X = X - X.mean(axis=0, keepdims=True)
    try:
        U, s, Vt = svd(X, full_matrices=False)
    except LinAlgError:
        U, s, Vt = svd(X, full_matrices=False, lapack_driver='gesvd')
    return U, s, Vt


def svd_pca(X, demean=True):
    U, s, Vt = safe_svd(X, demean=demean)
    return U * s[np.newaxis]
