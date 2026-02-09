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

def reliability_weighting_hyperalignment(X, Y, func, voxel_reliability_scores=None,threshold=None):
    """
    Hyperalignment with reliability weighting
    X, Y: (16, 285) - source and target data
    reliability_weights: (285,) - per-voxel reliability
    alpha: regularization strength
    """

    if voxel_reliability_scores is None:
       xfm = func(X, Y)
    else:
        if threshold is None:
            weights = np.clip(voxel_reliability_scores, 0.0, 1.0)
        else:
            weights = np.ones_like(voxel_reliability_scores)
            weights[voxel_reliability_scores < threshold] = 0.0

        # Apply reliability weighting to features (voxels)
        W_reliability = np.diag(np.sqrt(weights))  # (285, 285)

        # Transform X: X_scaled = X @ W_reliability
        X_scaled = X @ W_reliability  # (16, 285)

        # Standard ridge on scaled data
        R_scaled = func(X_scaled, Y)  # (285, 285)

        # Transform back to original voxel space
        xfm = W_reliability @ R_scaled  # (285, 285)
    return xfm
