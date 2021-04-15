import os
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))


def get_searchlights(lr, radius, tmpl='fsaverage', freq=32):
    npz_fn = f'{tmpl}_{lr}h_{radius}mm_freq{freq}_masked.npz'
    npz = np.load(npz_fn)
    sls = np.array_split(npz['concatenated'], npz['sections'])
    return sls
