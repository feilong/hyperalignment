import numpy as np
from scipy.stats import zscore
from scipy.linalg import LinAlgError
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

from hyperalignment.linalg import safe_svd
from hyperalignment.procrustes import procrustes


def PCA_decomposition(dss, max_npc=None, flavor='sklearn', adjust_ns=False, demean=True):
    """Decompose concatenated data matrices using PCA/SVD.

    Parameters
    ----------
    dss : ndarray of shape (ns, nt, nv)
    max_npc : integer or None
    flavor : {'sklearn', 'svd'}
    adjust_ns : bool
        Whether to adjust the variance of the output so that it doesn't increase with the number of subjects.
    demean : bool
        Whether to remove the mean of the columns prior to SVD.

    Returns
    -------
    XX : ndarray of shape (nt, npc)
    cc : ndarray of shape (npc, ns, nv)
    """
    ns, nt, nv = dss.shape
    X = dss.transpose(1, 0, 2).reshape(nt, ns * nv)
    if max_npc is not None:
        max_npc = min(max_npc, min(X.shape[0], X.shape[1]))
    if flavor == 'sklearn':
        try:
            if demean:
                pca = PCA(n_components=max_npc, random_state=0)
                XX = pca.fit_transform(X)
                cc = pca.components_.reshape(-1, ns, nv)
                if adjust_ns:
                    XX /= np.sqrt(ns)
                return XX, cc
            else:
                U, s, Vt = randomized_svd(X, (max_npc if max_npc is not None else min(X.shape)), random_state=0)
                if adjust_ns:
                    XX = U[:, :max_npc] * (s[np.newaxis, :max_npc] / np.sqrt(ns))
                else:
                    XX = U[:, :max_npc] * (s[np.newaxis, :max_npc])
                cc = Vt[:max_npc].reshape(-1, ns, nv)
                return XX, cc
        except LinAlgError as e:
            return PCA_decomposition(dss, max_npc=max_npc, flavor='svd', adjust_ns=adjust_ns, demean=demean)
    elif flavor == 'svd':
        U, s, Vt = safe_svd(X, demean=demean)
        if adjust_ns:
            XX = U[:, :max_npc] * (s[np.newaxis, :max_npc] / np.sqrt(ns))
        else:
            XX = U[:, :max_npc] * (s[np.newaxis, :max_npc])
        cc = Vt[:max_npc].reshape(-1, ns, nv)
        return XX, cc
    else:
        raise NotImplementedError


def compute_PCA_template(dss, sl=None, max_npc=None, flavor='sklearn', demean=True):
    if sl is not None:
        dss = dss[:, :, sl]
    XX, cc = PCA_decomposition(dss, max_npc=max_npc, flavor=flavor, adjust_ns=True, demean=demean)
    return XX


def compute_PCA_var1_template(dss, sl=None, max_npc=None, flavor='sklearn', demean=True):
    if sl is not None:
        dss = dss[:, :, sl]
    XX, cc = PCA_decomposition(dss, max_npc=max_npc, flavor=flavor, adjust_ns=False, demean=demean)
    w = np.sqrt(np.sum(cc**2, axis=2)).mean(axis=1)
    XX *= w[np.newaxis]
    return XX


def compute_PCA_var2_template(dss, sl=None, max_npc=None, flavor='sklearn', demean=True):
    if sl is not None:
        dss = dss[:, :, sl]
    XX, cc = PCA_decomposition(dss, max_npc=max_npc, flavor=flavor, adjust_ns=False, demean=demean)
    # w = np.sqrt(np.sum(cc**2, axis=2)).mean(axis=1)
    w = np.exp(0.5 * np.log(np.sum(cc**2, axis=2)).mean(axis=1))
    XX *= w[np.newaxis]
    return XX


def compute_procrustes_template(
        dss, sl=None, reflection=True, scaling=False, zscore_common=True, level2_iter=1, dss2=None, **kwargs):
    if sl is not None:
        dss = dss[:, :, sl]
    common_space = np.copy(dss[0])
    aligned_dss = [dss[0]]
    for ds in dss[1:]:
        T = procrustes(ds, common_space, reflection=reflection, scaling=scaling)
        aligned_ds = ds.dot(T)
        if zscore_common:
            aligned_ds = np.nan_to_num(zscore(aligned_ds, axis=0))
        aligned_dss.append(aligned_ds)
        common_space = (common_space + aligned_ds) * 0.5
        if zscore_common:
            common_space = np.nan_to_num(zscore(common_space, axis=0))

    aligned_dss2 = []
    for level2 in range(level2_iter):
        common_space = np.zeros_like(dss[0])
        for ds in aligned_dss:
            common_space += ds
        for i, ds in enumerate(dss):
            reference = (common_space - aligned_dss[i]) / float(len(dss) - 1)
            if zscore_common:
                reference = np.nan_to_num(zscore(reference, axis=0))
            T = procrustes(ds, reference, reflection=reflection, scaling=scaling)
            if level2 == level2_iter - 1 and dss2 is not None:
                aligned_dss2.append(dss2[i].dot(T))
            aligned_dss[i] = ds.dot(T)

    # common_space = np.zeros_like(dss[0])
    # for ds in aligned_dss:
    #     common_space += ds
    common_space = np.sum(aligned_dss, axis=0)
    if zscore_common:
        common_space = np.nan_to_num(zscore(common_space, axis=0))
    else:
        common_space /= float(len(dss))
    if dss2 is not None:
        common_space2 = np.zeros_like(dss2[0])
        for ds in aligned_dss2:
            common_space2 += ds
        if zscore_common:
            common_space2 = np.nan_to_num(zscore(common_space2, axis=0))
        else:
            common_space2 /= float(len(dss))
        return common_space, common_space2

    return common_space


def compute_template(dss, sl=None, kind='procrustes', max_npc=None, common_topography=False, demean=False):
    mapping = {
        'pca': compute_PCA_template,
        'pcav1': compute_PCA_var1_template,
        'pcav2': compute_PCA_var2_template,
        'cls': compute_procrustes_template,
    }
    if kind == 'procrustes':
        tmpl = compute_procrustes_template(dss=dss, sl=sl, reflection=True, scaling=False, zscore_common=True)
    elif kind in mapping:
        tmpl = mapping[kind](dss=dss, sl=sl, max_npc=max_npc, demean=demean)
    else:
        raise ValueError

    if common_topography:
        if sl is not None:
            dss = dss[:, :, sl]
        ns, nt, nv = dss.shape
        T = procrustes(np.tile(tmpl, (ns, 1)), dss.reshape(ns*nt, nv))
        tmpl = tmpl @ T
    return tmpl


if __name__ == '__main__':
    from scipy.stats import zscore
    from hyperalignment.linalg import svd_pca
    rng = np.random.default_rng(0)
    dss = rng.standard_normal((1, 200, 100))
    dss = zscore(dss, axis=1)
    dss = np.tile(dss, (10, 1, 1))
    XX0 = compute_PCA_template(dss)
    XX1 = compute_PCA_var1_template(dss)
    XX2 = compute_PCA_var2_template(dss)
    np.testing.assert_allclose(XX0, XX1, atol=1e-7)
    np.testing.assert_allclose(XX0, XX2, atol=1e-7)
    XX3 = compute_procrustes_template(dss)
    np.testing.assert_allclose(dss[0], XX3, atol=1e-7)
    XX3 = svd_pca(XX3)
    np.testing.assert_allclose(np.abs(XX0[:, :100]), np.abs(XX3), atol=1e-7)
