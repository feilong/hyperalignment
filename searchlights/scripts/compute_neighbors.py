# coding: utf-8
import os
import numpy as np
import nibabel as nib

from mvpa2.support.nibabel.surf import Surface

fs_dir = os.path.expanduser('~/singularity_home/freesurfer/subjects')

for lr in 'lr':
    coords, faces = nib.freesurfer.io.read_geometry(os.path.join(fs_dir, 'fsaverage5', 'surf', lr+'h.white'))
    surf = Surface(coords, faces)
    mask = np.load('../fsaverage_{lr}h_mask.npy'.format(lr=lr))
    mask = mask[:10242]
    mapping = np.cumsum(mask) - 1
    nbrs = np.full((10242, 6), -1, dtype=int)
    for center, d in surf.neighbors.items():
        n = list(d.keys())
        n = [_ for _ in n if mask[_]]
        nbrs[center, :len(n)] = mapping[n]
    nbrs = nbrs[mask]
    np.save('../fsaverage_{lr}h_icoorder5-masked_neighbors.npy'.format(lr=lr), nbrs)
