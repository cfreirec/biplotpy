import numpy as np
import biplotpy as bp


def test_r_pretty_cases():
    assert list(bp.r_pretty([10.4, 33.9], 4)) == [10, 15, 20, 25, 30, 35]
    assert list(bp.r_pretty([71.1, 472], 4)) == [0, 100, 200, 300, 400, 500]
    assert list(bp.r_pretty([-3.2, 7.7], 3)) == [-5, 0, 5, 10]


def test_hj_properties():
    X = bp.load_iris().iloc[:, :4].to_numpy()
    b = bp.hj_biplot(X, dim=4)
    # HJ: row and column markers in principal coordinates; qualities sum to 100 with all dims
    np.testing.assert_allclose(b.qualities("columns").iloc[:, -1], 100, atol=1e-8)
    np.testing.assert_allclose(b.col_contrib_to_factor.sum(0), 100, atol=1e-8)
    Xs = b.scaled_data
    np.testing.assert_allclose(b.row_coordinates, Xs @ b.V, atol=1e-10)


def test_jk_reconstruction_and_projection():
    X = bp.load_iris().iloc[:, :4].to_numpy()
    b = bp.jk_biplot(X, dim=4)
    np.testing.assert_allclose(b.row_coordinates @ b.col_coordinates.T, b.scaled_data, atol=1e-10)
    np.testing.assert_allclose(bp.project_rows(b, X[:5]), b.row_coordinates[:5], atol=1e-10)
    g = bp.gh_biplot(X, dim=2)
    np.testing.assert_allclose(bp.project_rows(g, X[:5]), g.row_coordinates[:5], atol=1e-10)


def test_calibrated_axis_predicts():
    X = bp.load_mtcars()[["mpg", "disp", "hp", "drat", "wt", "qsec"]].to_numpy()
    b = bp.jk_biplot(X, dim=2, scaling=5)
    pos = bp.calibrated_axis(b.col_coordinates[0], [1.0])
    assert abs(pos[0] @ b.col_coordinates[0] - 1.0) < 1e-12


def test_bootstrap_runs():
    X = bp.load_iris().iloc[:, :4].to_numpy()
    res = bp.pca_bootstrap(X, dim=2, B=60, seed=1)
    v0 = res["initial_svd"][2]
    assert np.all(np.sign(res["loadings"].mean(0)[np.abs(v0) > 0.2]) == np.sign(v0[np.abs(v0) > 0.2]))
    lo, hi = res["ci_percentile_eigenvalue"]
    assert np.all(lo <= hi)
