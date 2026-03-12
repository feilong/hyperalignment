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


def searchlight_hyperalignment(
    X, Y, sls, sls_Y=None, sl_func=None, mat0=None, weights=None,n_jobs=1
):
    """
    Searchlight hyperalignment.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        The first data matrix, to be aligned to ``Y``.
    Y : ndarray of shape (n_samples, n_features_Y)
        The second data matrix, the target for alignment.
    sls : list of ndarrays
        Each ndarray contains the indices of the vertices in the searchlight
        for the first data matrix.
    sls_Y : list of ndarrays or None, default=None
        Searchlight indices for the second data matrix, if different from
        ``sls``.
    sl_func : callable
        A function that takes two arrays and computes the transformation
        between them.
    mat0 : sparse matrix or None, default=None
        The sparse matrix to initialize the transformation, because changing
        sparse matrix sparsity is very slow. If None, initialize using a dense
        array of zeros.
    weights : list of ndarrays or None, default=None
        Weights for combining searchlight transformations. If None, simply add
        the transformations without weighting.

    Returns
    -------
    mat : sparse matrix or ndarray
        The resulting transformation matrix, where each row corresponds to a
        vertex in ``X`` and each column corresponds to a vertex in ``Y``.
    """
    if sls_Y is None:
        sls_Y = sls

    mat = np.zeros((X.shape[1], Y.shape[1])) if mat0 is None else mat0.copy()
    if n_jobs == 1:
        if weights is not None:
            for sl_X, sl_Y, w in zip(sls, sls_Y, weights):
                t = sl_func(X[:, sl_X], Y[:, sl_Y])
                mat[np.ix_(sl_X, sl_Y)] += t * w[np.newaxis]
        else:
            warnings.warn('Legacy, do not use this, use searchlight_weights(dists=None) to get uniform weights as input to this function instead')
            for sl_X, sl_Y in zip(sls, sls_Y):
                t = sl_func(X[:, sl_X], Y[:, sl_Y])
                mat[np.ix_(sl_X, sl_Y)] += t
    else:
        with Parallel(n_jobs=n_jobs, batch_size=1, verbose=1) as parallel:
            local_xfms = parallel(
                delayed(sl_func)(X[:, sl_X], Y[:, sl_Y])
                for sl_X, sl_Y in zip(sls, sls_Y)
            )
        if weights is not None:
            for t, sl_X, sl_Y, w in zip(local_xfms, sls, sls_Y, weights):
                mat[np.ix_(sl_X, sl_Y)] += t * w[np.newaxis]
        else:
            warnings.warn('Legacy, do not use this, use searchlight_weights(dists=None) to get uniform weights as input to this function instead')
            for t, sl_X, sl_Y in zip(local_xfms, sls, sls_Y):
                mat[np.ix_(sl_X, sl_Y)] += t
    return mat


def searchlight_procrustes(
    X, Y, sls, sls_Y=None, mat0=None, reflection=True, scaling=False, weights=None,**kwargs
):
    """
    Searchlight hyperalignment using orthogonal Procrustes.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        The first data matrix, to be aligned to ``Y``.
    Y : ndarray of shape (n_samples, n_features_Y)
        The second data matrix, the target for alignment.
    sls : list of ndarrays
        Each ndarray contains the indices of the vertices in the searchlight
        for the first data matrix.
    sls_Y : list of ndarrays or None, default=None
        Searchlight indices for the second data matrix, if different from
        ``sls``.
    mat0 : sparse matrix or None, default=None
        The sparse matrix to initialize the transformation, because changing
        sparse matrix sparsity is very slow. If None, initialize using a dense
        array of zeros.
    reflection : bool, default=True
        Whether allows reflection in the transformation (True) or not (False).
    scaling : bool, default=False
        Whether allows global scaling (True) or not (False).
    weights : list of ndarrays or None, default=None
        Weights for combining searchlight transformations. If None, simply add
        the transformations without weighting.

    Returns
    -------
    mat : sparse matrix or ndarray
        The resulting transformation matrix, where each row corresponds to a
        vertex in ``X`` and each column corresponds to a vertex in ``Y``.
    """
    sl_func = functools.partial(procrustes, reflection=reflection, scaling=scaling)
    xfm = searchlight_hyperalignment(
        X, Y, sls, sls_Y=sls_Y, sl_func=sl_func, mat0=mat0, weights=weights,**kwargs
    )
    return xfm


def searchlight_ridge(X, Y, sls, sls_Y=None, mat0=None, alpha=1e3, weights=None,**kwargs):
    """
    Searchlight hyperalignment using ridge regression.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        The first data matrix, to be aligned to ``Y``.
    Y : ndarray of shape (n_samples, n_features_Y)
        The second data matrix, the target for alignment.
    sls : list of ndarrays
        Each ndarray contains the indices of the vertices in the searchlight
        for the first data matrix.
    sls_Y : list of ndarrays or None, default=None
        Searchlight indices for the second data matrix, if different from
        ``sls``.
    mat0 : sparse matrix or None, default=None
        The sparse matrix to initialize the transformation, because changing
        sparse matrix sparsity is very slow. If None, initialize using a dense
        array of zeros.
    alpha : float, default=1e3
        The regularization parameter for ridge regression.
    weights : list of ndarrays or None, default=None
        Weights for combining searchlight transformations. If None, simply add
        the transformations without weighting.

    Returns
    -------
    mat : sparse matrix or ndarray
        The resulting transformation matrix, where each row corresponds to a
        vertex in ``X`` and each column corresponds to a vertex in ``Y``.
    """
    sl_func = functools.partial(ridge, alpha=alpha)
    xfm = searchlight_hyperalignment(
        X, Y, sls, sls_Y=sls_Y, sl_func=sl_func, mat0=mat0, weights=weights,**kwargs
    )
    return xfm


def searchlight_template(dms, sls, dists, radius, n_jobs=1, tpl_kind="pca"):
    weights = compute_searchlight_weights(sls, dists, radius)
    tpl = np.zeros_like(dms[0])

    if n_jobs == 1:
        for sl, w in zip(sls, weights):
            local_template = compute_template(
                dms, sl=sl, kind=tpl_kind, max_npc=len(sl), common_topography=True
            )
            tpl[:, sl] += local_template * w[np.newaxis]
    else:
        with Parallel(n_jobs=n_jobs, batch_size=1, verbose=1) as parallel:
            local_templates = parallel(
                delayed(compute_template)(
                    dms, sl=sl, kind=tpl_kind, max_npc=len(sl), common_topography=True
                )
                for sl in sls
            )

        for local_template, w, sl in zip(local_templates, weights, sls):
            tpl[:, sl] += local_template * w[np.newaxis]
    return tpl
