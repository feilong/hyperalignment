import numpy as np

from .procrustes import procrustes


def compute_searchlight_weights(sls, dists, radius):
    """
    weights = compute_searchlight_weights(sls, dists, radius)
    """
    nv = len(sls)
    weights_sum = np.zeros((nv, ))
    for sl, d in zip(sls, dists):
        w = (radius - d) / radius
        weights_sum[sl] += w
    # print(np.percentile(weights_sum, np.linspace(0, 100, 11)))
    weights = []
    for sl, d in zip(sls, dists):
        w = (radius - d) / radius
        w /= weights_sum[sl]
        weights.append(w)
    return weights


def searchlight_procrustes(X, Y, sls, dists, radius, T0=None, reflection=True, scaling=False):
    T = np.zeros((X.shape[1], Y.shape[1])) if T0 is None else T0.copy()
    weights = compute_searchlight_weights(sls, dists, radius)
    for sl, w in zip(sls, weights):
        t = procrustes(X[:, sl], Y[:, sl], reflection=reflection, scaling=scaling)
        T[np.ix_(sl, sl)] += t * w[np.newaxis]
    return T
