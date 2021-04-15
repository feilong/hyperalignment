import numpy as np

from .linalg import safe_svd


def procrustes(source, target, reflection=True, scaling=False, check_finite=True):
    A = target.T.dot(source).T
    U, s, Vt = safe_svd(A, demean=False)
    T = np.dot(U, Vt)
    if not reflection:
        sign = np.sign(np.linalg.det(T))
        s[-1] *= sign
        if sign < 0:
            T -= np.outer(U[:, -1], Vt[-1, :]) * 2
    if scaling:
        scale = s.sum() / (source.var(axis=0).sum() * source.shape[0])
        T *= scale
    return T


def searchlight_procrustes(X, Y, sls, reflection=True, scaling=False):
    counts = np.zeros((X.shape[1], ))
    T = np.zeros((X.shape[1], Y.shape[1]))
    for sl in sls:
        T[np.ix_(sl, sl)] += procrustes(X[:, sl], Y[:, sl], reflection=reflection, scaling=scaling)
        counts[sl] += 1
    return T, counts
