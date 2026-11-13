"""Phase 4: CA, MCA, CCA, RDA, PCoA, distances, MDS, external logistic, CDA,
rotations, FA, NIPALS, PLS, co-inertia, polychoric — against R
(fixtures/phase4.json from fixtures/make_phase4.R) — and every graphic."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import biplotpy as bp

F = json.loads((Path(__file__).parent / "fixtures" / "phase4.json").read_text())


def A(x):
    if isinstance(x, list) and x and isinstance(x[0], list):
        return np.array([[np.nan if v in ("NA", None) else v for v in r] for r in x], float)
    return np.array(x, float)


def align(P, R):
    P = A(P).copy(); R = A(R)
    for k in range(min(P.shape[1], R.shape[1])):
        if np.nansum(P[:, k] * R[:, k]) < 0:
            P[:, k] *= -1
    return P[:, :R.shape[1]]


def close(a, b, tol=1e-9):
    np.testing.assert_allclose(A(a), A(b), atol=tol, rtol=0)


IRIS = bp.load_iris()
X4, SP = IRIS.iloc[:, :4], IRIS["Species"]


# ---------------------------------------------------------------- correspondence
@pytest.mark.parametrize("alpha", [1, 0, 0.5])
def test_ca_matches_multbiplotr(alpha):
    f = F["ca"]; r = f[f"mbr_{alpha}"]
    m = bp.ca(A(f["N"]), alpha=alpha)
    close(align(m.row_coordinates, A(r["rows"])), r["rows"]); close(align(m.col_coordinates, A(r["cols"])), r["cols"])
    close(m.eigenvalues, A(r["eig"])[:len(m.eigenvalues)])
    close(m.row_qualities, r["rq"], 0.0051)          # MultBiplotR rounds to 2 decimals


def test_ca_matches_ca_package():
    p = F["ca"]["ca_pkg"]; m = bp.ca(A(F["ca"]["N"]))
    close(np.sqrt(m.eigenvalues), p["sv"]); close(align(m.row_standard, A(p["rowcoord"])), p["rowcoord"])
    close(align(m.col_standard, A(p["colcoord"])), p["colcoord"]); close(m.row_contributions / 100, p["ctr_r"])


def test_mca_matches_factominer():
    f = F["mca"]; m = bp.mca(pd.DataFrame(f["X"]), dim=3)
    close(m.eigenvalues[:len(f["eig"])], f["eig"]); close(align(m.row_coordinates, A(f["ind"])), f["ind"], 1e-12)
    idx = [m.col_names.index(n) for n in f["var_names"]]
    close(align(m.col_coordinates[idx], A(f["var"])), f["var"], 1e-12); close(m.eta2, f["eta2"], 1e-12)


def test_cca_matches_multbiplotr_and_vegan():
    f = F["cca"]; m = bp.cca(A(f["Y"]), A(f["Z"]), dim=3)
    close(m.eigenvalues, f["vegan_eig"]); close(m.total_inertia, f["vegan_tot"])
    for mine, key in ((m.col_coordinates, "species"), (m.row_coordinates, "sites"), (m.lc_site_scores, "lc"),
                      (m.env_coordinates, "env")):
        close(align(mine, A(f[key])), f[key], 1e-12)


@pytest.mark.parametrize("scaling", [1, 2, 3])
def test_rda_matches_vegan(scaling):
    f = F["rda"]; r = f[f"s{scaling}"]
    m = bp.rda(A(F["cca"]["Y"]), A(F["cca"]["Z"]), scaling=scaling)
    close(m.eigenvalues, f["eig"], 1e-10); close(m.ca_eigenvalues[:len(f["ca_eig"])], f["ca_eig"], 1e-10)
    for mine, key in ((m.col_coordinates, "species"), (m.row_coordinates, "sites"), (m.lc_site_scores, "lc"),
                      (m.env_coordinates, "bp")):
        close(align(mine, A(r[key])), r[key], 1e-10)
    s = f["scaled"]; m2 = bp.rda(A(F["cca"]["Y"]), A(F["cca"]["Z"]), scale=True)
    close(m2.eigenvalues, s["eig"], 1e-12); close(align(m2.col_coordinates, A(s["species"])), s["species"], 1e-12)


# ---------------------------------------------------------------- distances
def test_pcoa_and_binary_distances():
    f = F["pcoa"]
    p = bp.pcoa(bp.continuous_distances(A(f["X"])), dim=3)
    close(p.eigenvalues[:3], A(f["eig"])[:3]); close(align(p.row_coordinates, A(f["coords"])), f["coords"], 1e-12)
    close(align(p.row_coordinates, A(f["ade4_li"])), f["ade4_li"], 1e-12)
    close(p.stress1, f["stress1"]); close(p.rsq, f["rsq"])
    Bn = A(f["Bin"])
    close(bp.binary_distances(Bn, coefficient="Jaccard"), f["jaccard"], 1e-12)
    for cf, v in f["binary"].items():
        close(bp.binary_distances(Bn, coefficient=cf, transformation="sqrt(1-S)"), v, 1e-12)
    bpco = bp.pcoa(bp.binary_distances(Bn, coefficient="Jaccard"))
    close(align(bpco.row_coordinates, A(f["bin_coords"])), f["bin_coords"], 1e-12)


def test_external_logistic_matches_multbiplotr():
    f = F["pcoa"]
    e = bp.external_logistic_biplot(A(f["bin_coords"]), A(f["Bin"]), 0.2)
    close(e.col_parameters, f["ext_params"], 1e-10); close(e.var_info["Deviance"], f["ext_dev"], 1e-10)
    close(e.var_info["% correct"] / 100, f["ext_pct"])


@pytest.mark.parametrize("kind", ["ratio", "interval", "ordinal", "ordinal_s"])
def test_mds_matches_smacof(kind):
    f = F["mds"]; r = f[kind]
    m = bp.mds(A(f["D"]), type="ordinal" if kind.startswith("ordinal") else kind,
               ties="secondary" if kind == "ordinal_s" else "primary")
    assert m.niter == r["niter"]
    close(m.stress, r["stress"], 1e-12); close(align(m.row_coordinates, A(r["conf"])), r["conf"], 1e-10)


def test_canonical_distance_analysis_matches_multbiplotr():
    f = F["cda"]
    cd = bp.canonical_distance_analysis(bp.continuous_distances(X4.to_numpy()), SP, X=X4.to_numpy(), nperm=99, seed=1)
    close(np.squeeze(cd.F), np.squeeze(A(f["F"])), 1e-9); close(cd.TSS, f["TSS"]); close(np.squeeze(cd.BSS), np.squeeze(A(f["BSS"])))
    close(align(cd.group_coordinates, A(f["means"])), f["means"], 1e-12)
    close(align(cd.row_coordinates, A(f["rows"])), f["rows"], 1e-12)
    assert cd.p_value < 0.05


# ---------------------------------------------------------------- factor-type
def test_rotations_match_gparotation():
    f = F["fa"]; L0 = A(f["L0"])
    for k, v in f["rot"].items():
        close(bp.rotate(L0, k, gamma=0.5 if k == "oblimin" else 0)[0], v, 1e-8)
    r5 = f["rot5"]; L5 = A(r5["L"])
    close(bp.rotate(L5, "varimax")[0], r5["varimax"], 1e-8); close(bp.rotate(L5, "quartimin")[0], r5["quartimin"], 1e-8)
    close(bp.rotate(L5, "varimax", normalize=True)[0], r5["varimax_norm"], 1e-8)


def test_fa_biplot_matches_multbiplotr_and_psych():
    f = F["fa"]
    fb = bp.fa_biplot(X4, 2, "varimax")
    close(align(fb.col_coordinates, A(f["cols_vm"])), f["cols_vm"], 1e-10); close(fb.communalities, f["comm"], 1e-12)
    close(align(bp.fa_biplot(X4, 2, "varimax", scores="bartlett").row_coordinates, A(f["bartlett"])), f["bartlett"], 1e-10)
    close(align(fb.row_coordinates, A(f["thurstone"])), f["thurstone"], 1e-10)
    close(align(bp.fa_biplot(X4, 2, "none").col_coordinates, A(f["cols_none"])), f["cols_none"], 1e-10)


def test_nipals_with_missing_matches_ade4():
    f = F["nipals"]; nm = bp.nipals(A(f["Xmiss"]), 2)
    close(align(nm["li"], A(f["li"])), f["li"], 1e-10); close(align(nm["c1"], A(f["c1"])), f["c1"], 1e-10)
    close(nm["eig"], f["eig"], 1e-10); close(nm["rec"], f["rec"], 1e-10)
    full = bp.nipals_biplot(X4, 2, alpha=1); ref = bp.pca_biplot(X4, alpha=1, dim=2)
    close(align(full.row_coordinates, ref.row_coordinates), ref.row_coordinates, 1e-4)   # NIPALS stops at tol 1e-6 on the scores


def test_pls_matches_pls_package():
    f = F["pls"]; r = bp.pls_regression(A(f["X"]), A(f["Y"]), 3, True)
    for mine, key in ((r["scores"], "scores"), (r["loadings"], "loadings"), (r["y_loadings"], "yloadings"),
                      (r["weights"], "weights")):
        close(align(mine, A(f[key])), f[key], 1e-6)
    close(r["fitted"], f["fitted"], 1e-6); close(r["coef"], f["coef"], 1e-6)


def test_coinertia_matches_ade4():
    f = F["coin"]; c = bp.coinertia(A(f["X"]), A(f["Y"]), True, False)
    close(c["eig"], f["eig"], 1e-10); close(c["RV"], f["RV"], 1e-12)
    for k in ("c1", "l1", "co", "li", "lX", "lY"):
        close(align(c[k], A(f[k])), f[k], 1e-10)


def test_polychoric_matches_psych():
    f = F["poly"]; pc = bp.polychoric(A(f["X"]).astype(int))
    close(pc["tau"], f["tau"], 1e-12)
    close(pc["rho"], f["rho"], 1.2e-4)          # psych::polychoric uses optimize() with tol ~1.2e-4


def test_polychoric_biplot_recovers_latent_scores():
    rng = np.random.default_rng(2)
    n = 250; F_ = rng.normal(size=(n, 2)); L = rng.uniform(0.5, 0.85, (6, 2)) * np.array([[1, 0]] * 3 + [[0, 1]] * 3)
    lat = F_ @ L.T + rng.normal(size=(n, 6)) * np.sqrt(1 - (L ** 2).sum(1))
    Xo = np.digitize(lat, [-0.8, 0, 0.8]) + 1
    pb = bp.polychoric_biplot(Xo, 2)
    from scipy.linalg import orthogonal_procrustes
    R, _ = orthogonal_procrustes(pb.A, F_)
    # 3 four-category items per factor: the attainable correlation is about sqrt(reliability) ~ 0.75
    assert min(np.corrcoef((pb.A @ R)[:, k], F_[:, k])[0, 1] for k in range(2)) > 0.65


# ---------------------------------------------------------------- graphics (smoke tests)
def test_every_plot_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import runpy
    import sys
    gal = Path(__file__).parents[1] / "examples" / "gallery.py"
    runpy.run_path(str(gal), run_name="__main__")
    out = gal.parent / "gallery"
    assert len(list(out.glob("*.png"))) >= 40


def test_calibrated_axes_read_group_means_in_cva():
    """In a CVA with 3 groups the plane is exact for the group means, so each
    group marker read on the calibrated axis of a variable gives its mean."""
    cb = bp.canonical_biplot(X4, SP)
    J, H = cb.group_coordinates[:, :2], cb.col_coordinates[:, :2]
    means = pd.DataFrame(cb.scaled_data).groupby(np.asarray(cb.group)).mean().loc[cb.group_names].to_numpy()
    close(J @ H.T, means, 1e-10)


def test_overlays_have_their_coverage():
    from matplotlib.path import Path as MPath
    rng = np.random.default_rng(0)
    P = rng.multivariate_normal([0, 0], [[2, 0.8], [0.8, 1]], 4000)
    E = bp.concentration_ellipse(P, 0.95)
    assert abs(MPath(E).contains_points(P).mean() - 0.95) < 0.015
    bag = bp.alpha_bag(P[:500], 0.9)
    inside = MPath(bag).contains_points(P[:500], radius=1e-9).mean()
    assert 0.85 <= inside <= 0.95
