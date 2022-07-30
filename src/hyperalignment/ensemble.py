import numpy as np


def compute_ensemble_indices(nt, n_perms=10, n_folds=5, blocksize=4, buffersize=4, seed=0, mask=None):
    rng = np.random.default_rng(seed=seed)
    if mask is not None:
        nt = mask.shape[0]
    n_blocks = nt // blocksize
    remainder = nt % blocksize

    train_idx_li = []
    test_idx_li = []

    if mask is not None:
        mapping = np.cumsum(mask) - 1

    for i in range(n_perms):
        shift = rng.integers(remainder + 1)
        folds = np.array_split(rng.permutation(n_blocks), n_folds)
        for test_blocks in folds:
            test_idx = np.ravel(
                np.tile(test_blocks * blocksize, (blocksize, 1))
                + np.arange(blocksize)[:, np.newaxis] + shift)

            train_mask = np.ones((nt + buffersize * 2, ), dtype=bool)
            train_mask[test_idx + buffersize] = False
            for buffer in range(1, buffersize + 1):
                train_mask[test_idx + buffersize + buffer] = False
                train_mask[test_idx + buffersize - buffer] = False
            if mask is not None:
                train_mask[buffersize:nt+buffersize] = np.logical_and(mask, train_mask[buffersize:nt+buffersize])
            train_idx = np.where(train_mask[buffersize:nt+buffersize])[0]
            train_idx = rng.choice(train_idx, size=nt, replace=True)

            test_mask = np.ones((nt + buffersize*2, ), dtype=bool)
            idx = np.unique(train_idx)
            test_mask[idx + buffersize] = False
            for buffer in range(1, buffersize + 1):
                test_mask[idx + buffersize + buffer] = False
                test_mask[idx + buffersize - buffer] = False
            if mask is not None:
                test_mask[buffersize:nt+buffersize] = np.logical_and(mask, test_mask[buffersize:nt+buffersize])
            test_idx = np.where(test_mask[buffersize:nt+buffersize])[0]

            if mask is not None:
                train_idx = mapping[train_idx]
                test_idx = mapping[test_idx]

            train_idx_li.append(train_idx)
            test_idx_li.append(test_idx)

    return train_idx_li, test_idx_li
