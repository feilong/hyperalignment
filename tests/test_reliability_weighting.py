import numpy as np
from hyperalignment.reliability import reliability_weighting_hyperalignment,get_hyperalignment_func

n_samples = 20
n_features = 10

# dummy func: always returns identity (easy algebra check)
def identity_solver(X, Y):
    # expecting X: (samples x features)
    # returns transformation matrix (features x features)
    return np.eye(X.shape[1])

def random_data():
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    Y = X.copy()  # for some tests
    return X, Y

def test_no_weights_identity_output():
    X, Y = random_data()
    R = reliability_weighting_hyperalignment(X, Y, identity_solver)
    assert np.allclose(R, np.eye(n_features), atol=1e-6)

def test_all_ones_equal_no_weights():
    X, Y = random_data()
    R1 = reliability_weighting_hyperalignment(X, Y, identity_solver)
    R2 = reliability_weighting_hyperalignment(X, Y, identity_solver, voxel_reliability_scores=np.ones(n_features))
    assert np.allclose(R1, R2, atol=1e-6)

def test_all_zero_reliability():
    X, Y = random_data()
    R = reliability_weighting_hyperalignment(X, Y, identity_solver, voxel_reliability_scores=np.zeros(n_features))
    assert np.allclose(R, np.zeros((n_features, n_features)), atol=1e-6)

def test_threshold_zero_outs():
    X, Y = random_data()
    reliability = np.linspace(0, 1, n_features)
    R = reliability_weighting_hyperalignment(X, Y, identity_solver, voxel_reliability_scores=reliability, threshold=0.5)
    # find which columns should be zero
    zero_cols = np.where(reliability < 0.5)[0]
    col_norms = np.linalg.norm(R, axis=0)
    assert np.allclose(col_norms[zero_cols], 0, atol=1e-6)

def test_identity_algebra():
    X, Y = random_data()
    reliability = np.clip(np.random.rand(n_features),0,1)
    R = reliability_weighting_hyperalignment(X, Y, identity_solver, voxel_reliability_scores=reliability)

    W = np.diag(np.sqrt(reliability))
    # algebra says R should equal W since func returns identity
    assert np.allclose(R, W, atol=1e-6)

def test_procrustes_identity_case():
    X, Y = random_data()
    func = get_hyperalignment_func("procr")
    R = reliability_weighting_hyperalignment(X, Y, func)
    # procrustes may give slight floating error but should be close to identity
    assert np.allclose(R, np.eye(n_features), atol=1e-4)

def test_ridge_identity_case():
    X, Y = random_data()
    func = get_hyperalignment_func("ridge", alpha=1e-8)
    R = reliability_weighting_hyperalignment(X, Y, func)
    # ridge with identical data should also be near identity
    assert np.allclose(R, np.eye(n_features), atol=1e-4)

def test_reliability_synthetic():
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)

    # true reliability vector
    reliability = np.random.rand(n_features) # 0..1 random
    sqrtW = np.diag(np.sqrt(reliability))

    # true transform in scaled space
    U,_,Vt = np.linalg.svd(np.random.randn(n_features,n_features))
    R_real = U @ Vt       # nice orthogonal

    # construct the ground truth full transform
    xfm_real = sqrtW @ R_real

    # generate Y using that transform
    Y = X @ xfm_real

    # now estimate with YOUR function, we use procrustes solver
    func = get_hyperalignment_func("procr")

    xfm_est = reliability_weighting_hyperalignment(X, Y, func,
                                                   voxel_reliability_scores=reliability)

    # compare
    # handle global sign ambiguity
    if np.linalg.norm(xfm_est - xfm_real) > np.linalg.norm(xfm_est + xfm_real):
        xfm_real = -xfm_real

    assert np.allclose(xfm_est, xfm_real, atol=1e-2)

def test_reliability_synthetic_ridge():
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)

    # reliability vector (0..1 random)
    reliability = np.random.rand(n_features)
    sqrtW = np.diag(np.sqrt(reliability))

    # true linear transform in scaled space
    A = np.random.randn(n_features, n_features)   # ridge can solve for any linear mapping, not only orthogonal
                                    # (ridge is more general than procrustes)

    # full-transform ground truth
    xfm_real = sqrtW @ A

    # generate Y
    Y = X @ xfm_real

    # now estimate with YOUR code using ridge solver
    func = get_hyperalignment_func("ridge", alpha=1e-8)   # tiny alpha → LS

    xfm_est = reliability_weighting_hyperalignment(
        X, Y, func, voxel_reliability_scores=reliability
    )

    # ridge should recover original model extremely close because this is direct linear mapping
    assert np.allclose(xfm_est, xfm_real, atol=1e-2)