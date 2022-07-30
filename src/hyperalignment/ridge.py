import numpy as np
from datetime import datetime

from hyperalignment.linalg import safe_svd


def ridge_grid(X, y, alphas, npcs, train_idx=None):
    if train_idx is not None:
        X = X[train_idx]
        y = y[train_idx]
    # Assumes data have zero mean
    U, s, Vt = safe_svd(X, remove_mean=False)
    n_npcs = len(npcs)

    d = s[:, np.newaxis] / (alphas[np.newaxis, :] + (s**2)[:, np.newaxis])  # (k, n_alphas)
    UT_y = np.dot(U.T, y)  # (k, )
    d_UT_y = d * UT_y[..., np.newaxis]  # (k, n_alphas)
    betas = np.zeros((X.shape[1], len(alphas), n_npcs))

    npcs = [0] + list(npcs)
    for i in range(n_npcs):
        slc = slice(npcs[i], npcs[i+1])
        betas[..., i] = np.tensordot(Vt.T[:, slc], d_UT_y[slc], axes=(1, 0))

    betas = np.cumsum(betas, axis=2)

    return betas


def ensemble_ridge(X, y, alphas, npcs, train_idx_li, test_idx_li, msg=None):
    nt, nv = X.shape
    n_alphas, n_npcs = len(alphas), len(npcs)

    weights_test = np.zeros((nt, nv, n_alphas, n_npcs))
    count_test = np.zeros((nt, ), dtype=int)
    n_models = len(train_idx_li)
    weights = np.zeros((n_models, nv, n_alphas, n_npcs))
    count = 0

    for i, (train_idx, test_idx) in enumerate(zip(train_idx_li, test_idx_li)):
        betas = ridge_grid(X[train_idx], y[train_idx], alphas, npcs)
        weights_test[test_idx] += betas[np.newaxis]
        weights[i] = betas
        count_test[test_idx] += 1
        count += 1

    weights_test /= count_test[:, np.newaxis, np.newaxis, np.newaxis]
    weights = np.mean(weights, axis=0)

    pred = np.sum(X[:, :, np.newaxis, np.newaxis] * weights_test, axis=1)
    cost = np.sum((y[:, np.newaxis, np.newaxis] - pred)**2, axis=0)
    i, j = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
    alpha = alphas[i]
    npc = npcs[j]
    weights = weights[:, i, j]
    r2 = 1. - cost[i, j] / np.sum(y**2)
    if msg is not None:
        print(datetime.now(), msg, f'{r2:8.5f}, {np.log10(alpha):4.1f}', npc)
    return weights, pred[:, i, j], r2, alpha, npc
