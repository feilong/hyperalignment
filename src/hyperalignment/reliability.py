'''
Functions that are in testing phase
'''
import numpy as np
import functools
from hyperalignment.ridge import ridge
from hyperalignment.procrustes import procrustes

def get_hyperalignment_func(alignfunc,alpha=None):
    # which transformation method?
    if alignfunc == "procr" or alignfunc == "procrustes":
        func = functools.partial(procrustes, reflection=True, scaling=False)
    elif alignfunc == "ridge" or alignfunc == "ridgeCV":
        func = functools.partial(ridge, alpha=alpha)
    else:
        ValueError("Unsupported alignment method provided! Choose from 'procr','ridge'")
    return func

def reliability_weighting_hyperalignment(X, Y, func, voxel_reliability_scores=None, threshold=None):
    """
    Reliability-weighted hyperalignment.

    X, Y: (N, P)  # N timepoints/samples, P voxels/features
    func: callable (X, Y) -> transform
          - Procrustes: returns orthogonal Q (P,P) such that X @ Q ~ Y
          - Ridge: returns linear T_w (P,P) such that X_w @ T_w ~ Y_w

    voxel_reliability_scores: (P,) in [0,1] ideally (or will be clipped)
    threshold: if not None, binarize weights: w=1 if score>=threshold else 0

    Returns:
      - If func is procrustes-like: Q (P,P), orthogonal (up to numeric error)
      - If func is ridge-like:     T (P,P) mapped back to original space
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same shape, got {X.shape} vs {Y.shape}")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D arrays (N,P); got X.ndim={X.ndim}")

    P = X.shape[1]

    # No weighting
    if voxel_reliability_scores is None:
        return func(X, Y)

    scores = np.asarray(voxel_reliability_scores).reshape(-1)
    if scores.shape[0] != P:
        raise ValueError(f"voxel_reliability_scores must have length P={P}, got {scores.shape[0]}")

    if threshold is None:
        w = np.clip(scores.astype(float), 0.0, 1.0)
    else:
        w = np.ones(P, dtype=float)
        w[scores < threshold] = 0.0

    # Construct W = diag(sqrt(w))
    wsqrt = np.sqrt(w)
    # Efficient column scaling instead of making a dense diag
    X_scaled = X * wsqrt[None, :]
    Y_scaled = Y * wsqrt[None, :]

    T_scaled = func(X_scaled, Y_scaled)

    # Decide whether we should map back (ridge) or not (procrustes).
    # Procrustes returns an orthogonal matrix in voxel space; keep it orthogonal.
    # Ridge returns an unconstrained linear map in the scaled space; map back.
    name = getattr(func, "__name__", None)
    is_partial = hasattr(func, "keywords")  # functools.partial
    if is_partial:
        base = getattr(func, "func", None)
        base_name = getattr(base, "__name__", "")
    else:
        base_name = name or ""

    procrustes_like = "procr" in base_name.lower()

    if procrustes_like:
        # This is already the orthogonal Q that best aligns X_scaled -> Y_scaled.
        # Applying it to unweighted X is what you want, and it preserves orthogonality.
        return T_scaled

    # Ridge-like: map back to original space: T = W * T_scaled * W^{-1}
    # Use pseudo-inverse on W if any zeros
    inv_wsqrt = np.zeros_like(wsqrt)
    nz = wsqrt > 0
    inv_wsqrt[nz] = 1.0 / wsqrt[nz]

    # Left-multiply scales rows, right-multiply scales columns
    T = (wsqrt[:, None] * T_scaled) * inv_wsqrt[None, :]
    return T

def evaluate_transformation(X_val, Y_val, xfm):
    """
    Evaluate the transformation on validation set
    """
    # Predict on validation set
    Y_pred = X_val @ xfm  # (16,285) @ (285,285) = (16,285)

    # Calculate metrics
    mse = np.mean((Y_val - Y_pred) ** 2)
    correlation = np.mean([np.corrcoef(Y_val[:, i], Y_pred[:, i])[0, 1]
                          for i in range(Y_val.shape[1])])
    return {
        'mse': mse,
        'correlation': correlation
    }

def loo_cv(X, Y, func,voxel_reliability_scores=None):
    """
    Leave one run out cross-validation
    X : array (nruns, nT, nV)
    Y : array to align to (nT,nV)
    func: the function, ridge or procrustes used to align the data
    reliability: reliability [0-1] to be used for weighting
    """
    nruns = X.shape[0]
    mse_train_all = []
    mse_val_all   = []

    for i in range(nruns):
        X_val = X[i]
        idx = np.arange(nruns) != i
        X_train = np.mean(X[idx], axis=0)
        R = reliability_weighting_hyperalignment(X_train, Y, func, voxel_reliability_scores=voxel_reliability_scores,threshold=None)

        mt = evaluate_transformation(X_train, Y, R)
        mv = evaluate_transformation(X_val, Y, R)

        mse_train_all.append(mt['mse'])
        mse_val_all.append(mv['mse'])

    # average across LOO folds
    return np.mean(mse_train_all), np.mean(mse_val_all)