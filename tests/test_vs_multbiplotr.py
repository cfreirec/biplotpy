"""Equivalence with MultBiplotR 25.11.15 (fixtures from tests/fixtures/make_fixtures.R).

SVD axes are defined up to sign; each dimension is aligned with the R solution
before comparing. MultBiplotR rounds contributions to 2 decimals and inertia to 3.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import biplotpy as bp

FX = pathlib.Path(__file__).parent / "fixtures"
R = json.load(open(FX / "multbiplotr.json"))
IRIS = pd.read_csv(FX / "iris.csv")
MT = pd.read_csv(FX / "mtcars.csv")
X_IRIS = IRIS.iloc[:, :4].to_numpy(float)
X_MT = MT[["mpg", "disp", "hp", "drat", "wt", "qsec"]].to_numpy(float)


def signs(a, ref):
    s = np.sign(np.sum(np.asarray(a) * np.asarray(ref), axis=0))
    s[s == 0] = 1
    return s


CASES = {f"iris_s{s}_a{a}": (X_IRIS, a, 2, s) for s in (4, 5) for a in (0, 0.5, 1)}
CASES.update({"iris_HJ_s5": (X_IRIS, 2, 3, 5), "mtcars_HJ_s5": (X_MT, 2, 3, 5), "mtcars_s4_a1": (X_MT, 1, 3, 4),
              "mtcars_s2_a1": (X_MT, 1, 2, 2), "mtcars_s10_a1": (X_MT, 1, 2, 10), "mtcars_s3_a1": (X_MT, 1, 2, 3)})


@pytest.mark.parametrize("name", list(CASES))
def test_classical_biplots(name):
    X, alpha, dim, sc = CASES[name]
    r = R["pca"][name]
    b = bp.pca_biplot(X, alpha=alpha, dim=dim, scaling=sc)
    ra, rb = np.array(r["a"]), np.array(r["b"])
    s = signs(b.col_coordinates, rb)
    np.testing.assert_allclose(b.scaled_data, np.array(r["scaled"]), atol=1e-12)
    np.testing.assert_allclose(b.row_coordinates * s, ra, atol=1e-10)
    np.testing.assert_allclose(b.col_coordinates * s, rb, atol=1e-10)
    np.testing.assert_allclose(b.eigenvalues, np.array(r["ev"]), rtol=1e-10, atol=1e-10 * max(r["ev"]))
    np.testing.assert_allclose(b.inertia, np.array(r["inertia"]), atol=6e-4)
    assert abs(b.scale_factor - r["scf"]) < 1e-12
    np.testing.assert_allclose(b.structure * s, np.array(r["structure"]), atol=1e-10)
    np.testing.assert_allclose(b.row_contributions, np.array(r["rc"]), atol=6e-3)
    np.testing.assert_allclose(b.col_contributions, np.array(r["cc"]), atol=6e-3)


@pytest.mark.parametrize("t", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
def test_transforms(t):
    np.testing.assert_allclose(bp.initial_transform(X_MT, t)["X"], np.array(R["transforms"][str(t)]), atol=1e-10)


def test_supplementary_continuous_variables():
    b = bp.pca_biplot(X_IRIS[:, :3], alpha=1, dim=2, scaling=5)
    rb = R["pca"]["iris_s5_a1"]  # not the same biplot; align with AddContVars fixture via structure
    ac = bp.add_continuous_variables(b, X_IRIS[:, [3]], scaling=5)
    r = R["addcont"]
    s = signs(ac["structure"], np.array(r["structure"]).reshape(1, -1))
    np.testing.assert_allclose(ac["structure"] * s, np.array(r["structure"]).reshape(1, -1), atol=1e-10)
    np.testing.assert_allclose(ac["coordinates"] * s, np.array(r["coords"]).reshape(1, -1), atol=1e-10)
    np.testing.assert_allclose(ac["b0"], np.atleast_1d(r["b0"]), atol=1e-12)
    np.testing.assert_allclose(ac["R2"], np.atleast_1d(r["R2"]), atol=1e-10)


@pytest.mark.parametrize("sc,key", [(5, "scales5"), (4, "scales4")])
def test_calibrated_scales(sc, key):
    b = bp.pca_biplot(X_MT, alpha=1, dim=2, scaling=sc)
    out = bp.biplot_scales(b, nticks=4)
    for j in range(b.p):
        np.testing.assert_allclose(out["ticks"][j], np.array(R[key]["ticks"][j]), atol=1e-10)
        np.testing.assert_allclose(out["labels"][j], np.array(R[key]["labels"][j]), atol=1e-10)


def _check_canonical(cb, r, full=True):
    s = signs(cb.col_coordinates, np.array(r["H"]))
    np.testing.assert_allclose(cb.group_coordinates * s, np.array(r["J"]), atol=1e-10)
    np.testing.assert_allclose(cb.col_coordinates * s, np.array(r["H"]), atol=1e-10)
    np.testing.assert_allclose(cb.eigenvalues, np.array(r["ev"]), rtol=1e-10)
    assert abs(cb.wilks_f - r["wilksf"]) / r["wilksf"] < 1e-10
    np.testing.assert_allclose(cb.anovas.to_numpy(), np.array(r["anovas"]), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(cb.bonf_rad, np.array(r["bonfrad"]), rtol=1e-10)
    if full:
        np.testing.assert_allclose(cb.canonical_weights * s, np.array(r["B"]), atol=1e-10)
        np.testing.assert_allclose(cb.row_coordinates * s, np.array(r["V"]), atol=1e-10)
        np.testing.assert_allclose(cb.inertia, np.array(r["inertia"]), rtol=1e-10)
        for k in ("univrad", "multrad", "chisrad"):
            np.testing.assert_allclose(getattr(cb, k.replace("rad", "_rad")), np.array(r[k]), rtol=1e-10)
        np.testing.assert_allclose(cb.group_contributions, np.array(r["groupcontr"]), atol=1e-10)
        np.testing.assert_allclose(cb.col_contributions, np.array(r["colcontr"]), atol=1e-10)
        np.testing.assert_allclose(cb.qlr_vars, np.array(r["qlrvars"]), atol=1e-10)
        np.testing.assert_allclose(cb.row_contributions, np.array(r["rowcontr"]), atol=1e-10)
        np.testing.assert_allclose(cb.expl_total, np.array(r["expltotal"]), atol=1e-10)
        np.testing.assert_allclose(cb.structure * s, np.array(r["structure"]), atol=1e-10)


def test_canonical_iris():
    _check_canonical(bp.canonical_biplot(X_IRIS, IRIS["Species"], transform=5), R["canonical_iris"])


def test_canonical_mtcars():
    _check_canonical(bp.canonical_biplot(X_MT, MT["cyl"].astype(str), transform=4), R["canonical_mtcars"], full=False)


def test_manova_matches_r_summary_manova():
    cb = bp.canonical_biplot(X_IRIS, IRIS["Species"], transform=5)
    m = cb.manova
    # values from R: summary(manova(cbind(...) ~ Species), test = ...)
    np.testing.assert_allclose(m.loc["Wilks", ["Statistic", "F", "df1", "df2"]], [0.02343863, 199.1453, 8, 288], rtol=1e-6)
    np.testing.assert_allclose(m.loc["Pillai", ["Statistic", "F", "df1", "df2"]], [1.191899, 53.46649, 8, 290], rtol=1e-6)
    np.testing.assert_allclose(m.loc["Hotelling-Lawley", ["Statistic", "F", "df1", "df2"]], [32.47732, 580.5321, 8, 286], rtol=1e-6)
    np.testing.assert_allclose(m.loc["Roy", ["Statistic", "F", "df1", "df2"]], [32.19193, 1166.957, 4, 145], rtol=1e-6)


def test_lda_classification_iris():
    cb = bp.canonical_variate_analysis(X_IRIS, IRIS["Species"])
    assert abs(cb.percent_correct - 0.98) < 1e-12      # MASS::lda on iris: 147/150
    assert (cb.predict(X_IRIS) == cb.predicted).all()
