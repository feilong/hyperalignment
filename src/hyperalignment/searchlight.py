import functools

import numpy as np
import scipy.sparse as sparse
from joblib import Parallel, delayed

from hyperalignment.local_template import compute_template
from hyperalignment.procrustes import procrustes
from hyperalignment.ridge import ridge


def compute_searchlight_weights(sls, dists=None, radius=None, return_sparse=False):
    """
    Compute the weights used for combining searchlight models.
    If `dists` is None, uniform weights are used.
    If `dists` is provided, distance-based weights are computed, where the
    weight for each vertex is highest at the center of the searchlight and
    decreases linearly to zero at the edge of the searchlight.

    Typical usage:
    >>> weights = compute_searchlight_weights(sls, dists, radius)
    >>> mat = compute_searchlight_weights(
    ...     sls, dists, radius, return_sparse=True)

    Parameters
    ----------
    sls : list of ndarrays
        Searchlight indices, where each ndarray contains the indices of the
        vertices in the searchlight.
    dists : list of ndarrays or None, default=None
        Each array contains the distances of the vertices in the searchlight
        to the center of the searchlight, used for distance-based weighting.
        If None, uniform weights are used.
    radius : float or None, default=None
        The radius of the searchlight, used for distance-based weighting.

    Returns
    -------
    weights : list of ndarrays, optional
        Each ndarray contains the weights for the vertices in the corresponding
        searchlight. The weights sum to 1 for each vertex. Returned only if
        `return_sparse` is False.
    mat : sparse matrix, optional
        A sparse matrix where each row corresponds to a searchlight and each
        column corresponds to a vertex. Returned only if `return_sparse` is
        True.
    """
    nv = int(np.concatenate(sls).max()) + 1
    weights_sum = np.zeros((nv,))
    weights = []

    if dists is None:
        for sl in sls:
            weights_sum[sl] += 1

        for sl in sls:
            w = 1.0 / weights_sum[sl]
            weights.append(w)

    else:
        # If dists is provided, use distance-based weighting
        assert radius is not None, "Radius must be provided if distances are given."
        for sl, d in zip(sls, dists):
            w = (radius - d) / radius
            weights_sum[sl] += w

        for sl, d in zip(sls, dists):
            w = (radius - d) / radius
            w /= weights_sum[sl]
            weights.append(w)

    if return_sparse:
        mat = sparse.lil_array((len(sls), nv))
        for i, w in enumerate(weights):
            mat[i, sls[i]] = w
        mat = mat.tocsr()
        return mat

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


def searchlight_procrustes(
    X, Y, sls, dists, radius, T0=None, reflection=True, scaling=False, weighted=True
):
    sl_func = functools.partial(procrustes, reflection=reflection, scaling=scaling)
    T = searchlight_hyperalignment(
        X, Y, sls, dists, radius, T0=T0, sl_func=sl_func, weighted=weighted
    )
    return T


def searchlight_ridge(X, Y, sls, dists, radius, T0=None, alpha=1e3, weighted=True):
    sl_func = functools.partial(ridge, alpha=alpha)
    T = searchlight_hyperalignment(
        X, Y, sls, dists, radius, T0=T0, sl_func=sl_func, weighted=weighted
    )
    return T


def searchlight_template(dss, sls, dists, radius, n_jobs=1, tmpl_kind="pca", weighted=True):

    tmpl = np.zeros_like(dss[0])
    if weighted:
            weights = compute_searchlight_weights(sls, dists, radius)

    if n_jobs == 1:
        if weighted:
            for sl, w in zip(sls, weights):
                local_template = compute_template(
                    dss, sl=sl, kind=tmpl_kind, max_npc=len(sl), common_topography=True
                )
                tmpl[:, sl] += local_template * w[np.newaxis]
        else:
            for sl in sls:
                local_template = compute_template(
                    dss, sl=sl, kind=tmpl_kind, max_npc=len(sl), common_topography=True
                )
                tmpl[:, sl] += local_template

    else:
        with Parallel(n_jobs=n_jobs, batch_size=1, verbose=1) as parallel:
            local_templates = parallel(
                delayed(compute_template)(
                    dss, sl=sl, kind=tmpl_kind, max_npc=len(sl), common_topography=True
                )
                for sl in sls
            )
        if weighted:
            for local_template, w, sl in zip(local_templates, weights, sls):
                tmpl[:, sl] += local_template * w[np.newaxis]
        else:
            for local_template, sl in zip(local_templates, sls):
                tmpl[:, sl] += local_template
    return tmpl
