import numpy as np
import functools
from joblib import Parallel, delayed

from hyperalignment.procrustes import procrustes
from hyperalignment.ridge import ridge
from hyperalignment.local_template import compute_template


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


def searchlight_hyperalignment(X, Y, sls, dists, radius, T0, sl_func, weighted=True):
    T = np.zeros((X.shape[1], Y.shape[1])) if T0 is None else T0.copy()
    if weighted:
        weights = compute_searchlight_weights(sls, dists, radius)
        for sl, w in zip(sls, weights):
            t = sl_func(X[:, sl], Y[:, sl])
            T[np.ix_(sl, sl)] += t * w[np.newaxis]
    else:
        for sl in sls:
            t = sl_func(X[:, sl], Y[:, sl])
            T[np.ix_(sl, sl)] += t
    return T


def searchlight_procrustes(X, Y, sls, dists, radius, T0=None, reflection=True, scaling=False, weighted=True):
    sl_func = functools.partial(procrustes, reflection=reflection, scaling=scaling)
    T = searchlight_hyperalignment(
        X, Y, sls, dists, radius, T0=T0, sl_func=sl_func, weighted=weighted)
    return T


def searchlight_ridge(X, Y, sls, dists, radius, T0=None, alpha=1e3, weighted=True):
    sl_func = functools.partial(ridge, alpha=alpha)
    T = searchlight_hyperalignment(
        X, Y, sls, dists, radius, T0=T0, sl_func=sl_func, weighted=weighted)
    return T


def searchlight_template(dss, sls, dists, radius, n_jobs=1, tmpl_kind='pca'):
    weights = compute_searchlight_weights(sls, dists, radius)
    with Parallel(n_jobs=n_jobs, batch_size=1, verbose=1) as parallel:
        local_templates = parallel(
            delayed(compute_template)(dss, sl=sl, kind=tmpl_kind, max_npc=len(sl), common_topography=True)
            for sl in sls)

    tmpl = np.zeros_like(dss[0])
    for local_template, w, sl in zip(local_templates, weights, sls):
        tmpl[:, sl] += local_template * w[np.newaxis]
    return tmpl
