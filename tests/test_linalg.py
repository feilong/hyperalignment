import numpy as np
import scipy.linalg as linalg
from sklearn.decomposition import PCA
from hyperalignment.linalg import safe_svd, svd_pca


class TestSafeSVD:
    def test_tall_matrix(self):
        rng = np.random.default_rng(0)
        for rep in range(3):
            mat = rng.standard_normal((200, 100))

            U1, s1, Vt1 = np.linalg.svd(mat, full_matrices=False, compute_uv=True)
            U2, s2, Vt2 = linalg.svd(mat, full_matrices=False, compute_uv=True)
            U3, s3, Vt3 = safe_svd(mat, remove_mean=False)

            # Different SVD methods should generate the same results.
            for uu in [(U1, U2), (U2, U3), (U1, U3)]:
                np.testing.assert_allclose(uu[0], uu[1])
            for vv in [(Vt1, Vt2), (Vt2, Vt3), (Vt1, Vt3)]:
                np.testing.assert_allclose(vv[0], vv[1])
            for ss in [(s1, s2), (s1, s3), (s2, s3)]:
                np.testing.assert_allclose(ss[0], ss[1])

            # Re-generate the original matrix by matrix multiplications.
            np.testing.assert_allclose(mat, U3 @ np.diag(s3) @ Vt3)
            np.testing.assert_allclose(mat, (U3 * s3[np.newaxis]) @ Vt3)


            # Test with mean removal
            U3, s3, Vt3 = safe_svd(mat, remove_mean=True)
            mat1 = mat - mat.mean(axis=0, keepdims=True)
            U1, s1, Vt1 = np.linalg.svd(mat1, full_matrices=False, compute_uv=True)
            U2, s2, Vt2 = linalg.svd(mat1, full_matrices=False, compute_uv=True)

            for uu in [(U1, U2), (U2, U3), (U1, U3)]:
                np.testing.assert_allclose(uu[0], uu[1])
            for vv in [(Vt1, Vt2), (Vt2, Vt3), (Vt1, Vt3)]:
                np.testing.assert_allclose(vv[0], vv[1])
            for ss in [(s1, s2), (s1, s3), (s2, s3)]:
                np.testing.assert_allclose(ss[0], ss[1])

            np.testing.assert_allclose(mat1, U3 @ np.diag(s3) @ Vt3)
            np.testing.assert_allclose(mat1, (U3 * s3[np.newaxis]) @ Vt3)

    def test_wide_matrix(self):
        rng = np.random.default_rng(0)
        for rep in range(3):
            mat = rng.standard_normal((100, 200))

            U1, s1, Vt1 = np.linalg.svd(mat, full_matrices=False, compute_uv=True)
            U2, s2, Vt2 = linalg.svd(mat, full_matrices=False, compute_uv=True)
            U3, s3, Vt3 = safe_svd(mat, remove_mean=False)

            for uu in [(U1, U2), (U2, U3), (U1, U3)]:
                np.testing.assert_allclose(uu[0], uu[1])
            for vv in [(Vt1, Vt2), (Vt2, Vt3), (Vt1, Vt3)]:
                np.testing.assert_allclose(vv[0], vv[1])
            for ss in [(s1, s2), (s1, s3), (s2, s3)]:
                np.testing.assert_allclose(ss[0], ss[1])

            np.testing.assert_allclose(mat, U3 @ np.diag(s3) @ Vt3)
            np.testing.assert_allclose(mat, (U3 * s3[np.newaxis]) @ Vt3)

            U3, s3, Vt3 = safe_svd(mat, remove_mean=True)
            mat1 = mat - mat.mean(axis=0, keepdims=True)
            U1, s1, Vt1 = np.linalg.svd(mat1, full_matrices=False, compute_uv=True)
            U2, s2, Vt2 = linalg.svd(mat1, full_matrices=False, compute_uv=True)

            for uu in [(U1, U2), (U2, U3), (U1, U3)]:
                np.testing.assert_allclose(uu[0], uu[1])
            for vv in [(Vt1, Vt2), (Vt2, Vt3), (Vt1, Vt3)]:
                np.testing.assert_allclose(vv[0], vv[1])
            for ss in [(s1, s2), (s1, s3), (s2, s3)]:
                np.testing.assert_allclose(ss[0], ss[1])

            np.testing.assert_allclose(mat1, U3 @ np.diag(s3) @ Vt3)
            np.testing.assert_allclose(mat1, (U3 * s3[np.newaxis]) @ Vt3)


class Test_SVD_PCA:
    def test_tall_matrix(self):
        rng = np.random.default_rng(0)
        for rep in range(3):
            mat = rng.standard_normal((200, 100))

            pca = PCA()
            X1 = pca.fit_transform(mat)
            X2 = svd_pca(mat)

            # PCA results are not unique -- the sign of a PC can be
            # flipped
            err1 = np.abs(X1 - X2)
            err2 = np.abs(X1 + X2)

            np.testing.assert_allclose(
                np.minimum(err1, err2),
                np.zeros_like(X1),
                atol=1e-10)

    def test_wide_matrix(self):
        rng = np.random.default_rng(0)
        for rep in range(3):
            mat = rng.standard_normal((100, 200))

            pca = PCA()
            X1 = pca.fit_transform(mat)
            X2 = svd_pca(mat)

            err1 = np.abs(X1 - X2)
            err2 = np.abs(X1 + X2)

            np.testing.assert_allclose(
                np.minimum(err1, err2),
                np.zeros_like(X1),
                atol=1e-10)
