import os
import numpy as np


for radius in [20, 15, 13, 10, 7]:
    for lr in 'lr':
        fn1 = f'../fsaverage_{lr}h_{radius}mm_icoorder5.npz'
        fn2 = f'../fsaverage_{lr}h_{radius}mm_freq32_masked.npz'
        if not all([os.path.exists(_) for _ in [fn1, fn2]]):
            continue
        npz1 = np.load(fn1)
        npz2 = np.load(fn2)

        np.testing.assert_array_equal(npz1['sections'], npz2['sections'])
        sls1 = np.array_split(npz1['concatenated'], npz1['sections'])
        sls2 = np.array_split(npz2['concatenated'], npz2['sections'])

        for sl1, sl2 in zip(sls1, sls2):
            np.testing.assert_array_equal(np.sort(sl1), sl2)
        dists = np.array_split(npz1['concatenated_dists'], npz1['sections'])
        for d in dists:
            np.testing.assert_array_equal(np.sort(d), d)
        print(lr, radius)
