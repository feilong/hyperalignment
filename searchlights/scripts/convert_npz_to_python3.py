import os
from glob import glob
import numpy as np


def convert_old_npy_files():
    for fn in sorted(glob('*.npz')):
        out_fn = os.path.join('..', fn)
        npz = np.load(fn, allow_pickle=True, encoding='bytes')
        sls, dists = npz['sls'], npz['dists']
        lengths = []
        out_sls, out_dists = [], []
        for sl, d in zip(sls, dists):
            lengths.append(len(sl))
            sort_idx = np.argsort(d)
            out_sls.append(sl[sort_idx])
            out_dists.append(d[sort_idx])
        sections = np.cumsum(lengths)[:-1]
        concatenated = np.concatenate(out_sls)
        concatenated_dists = np.concatenate(out_dists)
        np.savez(out_fn, concatenated=concatenated, concatenated_dists=concatenated_dists, sections=sections)


if __name__ == '__main__':
    convert_old_npy_files()
