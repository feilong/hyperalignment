import os
import numpy as np
import nibabel as nib

from mvpa2.support.nibabel.surf import Surface

fs_dir = os.path.expanduser('~/singularity_home/freesurfer/subjects/fsaverage')


def compute_cortical_mask(lr):
    fn = '{fs_dir}/label/{lr}h.aparc.annot'.format(fs_dir=fs_dir, lr=lr)
    labels, ctab, names = nib.freesurfer.io.read_annot(fn)
    mask = (labels != -1)
    assert np.sum(mask[:10242]) == {'l': 9372, 'r': 9370}[lr]
    return mask


def compute_searchlights(lr, radius, icoorder=5, overwrite=False):
    out_fn = 'fsaverage_{lr}h_{radius}mm_icoorder{icoorder}.npz'.format(**locals())
    if os.path.exists(out_fn) and not overwrite:
        return

    coords1, faces1 = nib.freesurfer.io.read_geometry(os.path.join(fs_dir, 'surf', lr+'h.white'))
    coords2, faces2 = nib.freesurfer.io.read_geometry(os.path.join(fs_dir, 'surf', lr+'h.pial'))
    np.testing.assert_array_equal(faces1, faces2)
    coords = (coords1.astype(np.float) + coords2.astype(np.float)) * 0.5
    surf = Surface(coords, faces1)
    nv = 4**icoorder * 10 + 2

    sls = []
    dists = []
    mask = compute_cortical_mask(lr)[:nv]
    cortical_indices = np.where(mask)[0]
    mapping = np.cumsum(mask) - 1
    # print(cortical_indices.shape)

    for center in cortical_indices:
        pairs = surf.dijkstra_distance(center, maxdistance=radius).items()
        neighbors = np.array([_[0] for _ in pairs])
        d = np.array([_[1] for _ in pairs])
        mask = np.in1d(neighbors, cortical_indices)
        neighbors = mapping[neighbors[mask]]
        d = d[mask]
        sls.append(neighbors)
        dists.append(d)
    np.savez(out_fn, sls=sls, dists=dists)
    print(out_fn, len(sls))


if __name__ == '__main__':
    for radius in [20, 15, 13, 10, 7]:
        for lr in 'lr':
            # compute_cortical_mask(lr)
            compute_searchlights(lr, radius)
