import warnings

import neuroboros as nb
import numpy as np
import pytest
from scipy import sparse

from hyperalignment.sparse import initialize_sparse_matrix


def reference(sls, nv):
    """The original lil_matrix-based construction."""
    mat = sparse.lil_matrix((nv, nv))
    for sl in sls:
        mat[np.ix_(sl, sl)] = 1.0
    mat = mat.tocsc()
    mat.data = np.zeros_like(mat.data)
    return mat


class TestInitializeSparseMatrix:
    def test_same_structure_as_reference(self):
        sls = nb.sls("l", 10)
        nv = int(np.concatenate(sls).max()) + 1
        mat = initialize_sparse_matrix(sls)
        ref = reference(sls, nv)

        assert mat.format == "csc"
        assert mat.shape == ref.shape == (nv, nv)
        assert mat.nnz == ref.nnz
        np.testing.assert_array_equal(mat.indptr, ref.indptr)
        np.testing.assert_array_equal(mat.indices, ref.indices)
        assert mat.has_canonical_format
        assert mat.dtype == np.float64
        assert not mat.data.any()

    def test_nv_and_dtype(self):
        sls = nb.sls("l", 10)
        nv = int(np.concatenate(sls).max()) + 100
        mat = initialize_sparse_matrix(sls, nv=nv, dtype=np.float32)
        assert mat.shape == (nv, nv)
        assert mat.dtype == np.float32

    def test_block_assignment_keeps_sparsity(self):
        sls = nb.sls("l", 10)
        mat = initialize_sparse_matrix(sls)
        nnz = mat.nnz
        with warnings.catch_warnings():
            warnings.simplefilter("error", sparse.SparseEfficiencyWarning)
            for sl in sls[:10]:
                mat[np.ix_(sl, sl)] += np.ones((len(sl), len(sl)))
        assert mat.nnz == nnz

    def test_cache(self, tmp_path):
        sls = nb.sls("l", 10)
        cache_fn = str(tmp_path / "sub" / "mat0.npz")
        mat = initialize_sparse_matrix(sls, cache_fn=cache_fn)
        cached = initialize_sparse_matrix(sls, cache_fn=cache_fn)
        assert cached.format == "csc"
        assert cached.shape == mat.shape
        np.testing.assert_array_equal(cached.indptr, mat.indptr)
        np.testing.assert_array_equal(cached.indices, mat.indices)

    def test_out_of_range_indices(self):
        with pytest.raises(IndexError):
            initialize_sparse_matrix([np.array([0, 3])], nv=3)
        with pytest.raises(IndexError):
            initialize_sparse_matrix([np.array([-1, 0])], nv=3)

    def test_empty_searchlights(self):
        mat = initialize_sparse_matrix([], nv=3)
        assert mat.format == "csc"
        assert mat.shape == (3, 3)
        assert mat.nnz == 0
        with pytest.raises(ValueError):
            initialize_sparse_matrix([])
