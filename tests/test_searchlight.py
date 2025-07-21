import neuroboros as nb
import numpy as np

from hyperalignment.searchlight import compute_searchlight_weights


class TestSearchlightWeights:
    def test_uniform_weights(self):
        sls = nb.sls("l", 20)
        weights = compute_searchlight_weights(sls)

        nv = int(np.concatenate(sls).max()) + 1
        weights_sum = np.zeros((nv,))

        for sl, w in zip(sls, weights):
            weights_sum[sl] += w

        np.testing.assert_allclose(weights_sum, 1.0)

    def test_distance_weights(self):
        radius = 20
        sls, dists = nb.sls("l", radius, return_dists=True)
        weights = compute_searchlight_weights(sls, dists, radius)

        nv = int(np.concatenate(sls).max()) + 1
        weights_sum = np.zeros((nv,))

        for sl, w in zip(sls, weights):
            weights_sum[sl] += w

        np.testing.assert_allclose(weights_sum, 1.0)

    def test_sparse_searchlights(self):
        sls = nb.sls("l", 20, "onavg-ico64", center_space="onavg-ico32")
        weights = compute_searchlight_weights(sls)

        nv = int(np.concatenate(sls).max()) + 1
        weights_sum = np.zeros((nv,))

        for sl, w in zip(sls, weights):
            weights_sum[sl] += w

        np.testing.assert_allclose(weights_sum, 1.0)
