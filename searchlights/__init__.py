import os
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))


def get_searchlights(lr, radius, tmpl='fsaverage', freq=32):
    npz_fn = f'{DIR}/{tmpl}_{lr}h_{radius}mm_freq{freq}_masked.npz'
    npz = np.load(npz_fn)
    sls = np.array_split(npz['concatenated'], npz['sections'])
    return sls


if __name__ == '__main__':
    for lr in 'lr':
        for radius in [10, 13, 15, 20]:
            sls = get_searchlights(lr, radius, 'fsaverage5', 32)
            print(lr, radius, len(sls), np.percentile([len(_) for _ in sls], np.linspace(0, 100, 11)))
