"""Phase 5 (biplotpy 0.4.0) against R: fixtures from tests/fixtures/make_phase5.R
(MultBiplotR 25.11.15, smacof 2.1-7, car)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import biplotpy as bp

F = json.loads((Path(__file__).parent / "fixtures" / "phase5.json").read_text())
D = F["data"]
X, Yb, Yo, Xb = (np.array(D[k], float) for k in ("X", "Yb", "Yo", "Xb"))


def _a(x):
    return np.array(list(x.values()) if isinstance(x, dict) else x, float)


def _signs(A, R):
    s = np.sign((A * R).sum(0)); s[s == 0] = 1
    return s


def close(a, b, tol=1e-9):
    assert np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))) < tol


# ----------------------------------------------------------------------------- constrained binary
def test_constrained_binary_logistic_biplot():
    cb = bp.constrained_logistic_biplot(Yb, X, penalization=0.1)
    f = F["cbin"]
    close(cb.betas.to_numpy(), f["Betas"])
    s = _signs(cb.A, _a(f["A"]))
    close(cb.A * s, f["A"])
    B = cb.B.copy(); B[:, 1:] *= s
    close(B, f["B"])
    for mine, ref in [(cb.null_deviances, "NullDev"), (cb.model_deviances, "ModelDev"), (cb.cox_snell, "CoxSnell"),
                      (cb.nagelkerke, "Nagelkerke"), (cb.mcfadden, "MacFaden"), (cb.communalities, "Communalities"),
                      (cb.sensitivity, "Sens"), (cb.specificity, "Spec")]:
        close(mine, f[ref])
    close(cb.loadings * s, f["Loadings"])
    sv = cb.sup_variables
    close(sv["b0"], f["sv_b0"]); close(sv["coordinates"] * s, f["sv_coords"]); close(sv["R2"], f["sv_R2"])


# ----------------------------------------------------------------------------- constrained / external ordinal
def test_constrained_ordinal_logistic_biplot():
    co = bp.constrained_ordinal_logistic_biplot(Yo, X, penalization=0.1)
    f = F["cord"]
    s = _signs(co.A, _a(f["A"]))
    close(co.A * s, f["A"]); close(co.B * s, f["coef"])
    cp = co.column_parameters
    close(cp["thresholds"].to_numpy(), f["thr"]); close(cp["fit"].to_numpy(), f["fit"], 1e-8)
    close(co.loadings * s, f["loadings"]); close(co.communalities, f["Communalities"])
    for k, r in [("deviance", "Deviance"), ("null_deviance", "DevianceNull"), ("df", "df"), ("AIC", "AIC"), ("BIC", "BIC")]:
        assert abs(getattr(co, k) - f[r]) < 1e-8


def test_external_ordinal_logistic_biplot():
    f = F["eord"]
    eo = bp.external_ordinal_logistic_biplot(_a(f["A"]), Yo)
    close(eo.B, f["coef"]); close(eo.column_parameters["thresholds"].to_numpy(), f["thr"])
    close(eo.column_parameters["fit"].to_numpy(), f["fit"], 1e-8); close(eo.loadings, f["loadings"])
    assert abs(eo.deviance - f["Deviance"]) < 1e-8


def test_ordinal_inputs_as_categoricals():
    Ycat = pd.DataFrame({f"o{j + 1}": pd.Categorical(Yo[:, j].astype(int), categories=[1, 2, 3, 4], ordered=True)
                         for j in range(Yo.shape[1])})
    a = bp.external_ordinal_logistic_biplot(_a(F["eord"]["A"]), Ycat)
    close(a.B, F["eord"]["coef"])


# ----------------------------------------------------------------------------- PLS with binary responses
def test_plsr1_bin():
    r = bp.plsr1_bin(Yb[:, 0], X, S=2)
    f = F["plsr1bin"]
    close(r["scores"], f["T"]); close(r["weights"], f["W"]); close(r["loadings"], f["P"])
    close(r["fit"]["beta"], f["beta"]); close(r["structure"], f["structure"])
    assert abs(r["fit"]["deviance"] - f["Deviance"]) < 1e-9


def test_binary_plsr_same_path_as_r():
    """Same random starts (R's generator, seed 0) and the same CG optimizer; the
    alternating CG path amplifies last-digit differences of the arithmetic, so
    the agreement is about 1e-4 (fitted probabilities 1e-5)."""
    b = bp.binary_plsr(Yb[:, :3], Xb, S=2, penalization=0.1, seed=0)
    f = F["binpls"]
    close(b["x_intercepts"], f["p0"], 1e-10); close(b["y_intercepts"], f["q0"], 1e-10)
    for k, r in [("x_scores", "TT"), ("x_loadings", "P"), ("y_scores", "U"), ("y_loadings", "Q")]:
        close(b[k], f[r], 1e-4)
    close(b["expected"], f["Expected"], 1e-5)
    assert abs(b["percent_correct"] - f["PercentCorrect"]) < 1e-12


def test_optim_cg_matches_r_rosenbrock():
    fr = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    gr = lambda x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
    r = bp.optim_cg(fr, gr, [-1.2, 1])
    # optim(c(-1.2, 1), fr, grr, method = "CG") in R 4.3.3
    close(r["par"], [-0.7648372957330416, 0.59275883367186466], 1e-12)
    assert r["counts"] == (402, 101) and r["convergence"] == 1


# ----------------------------------------------------------------------------- log-frequency biplot
def test_log_frequency_biplot():
    f = F["logfreq"]
    b = bp.log_frequency_biplot(np.array(F["logfreq_N"]), seed=11)
    s = _signs(b.row_coordinates, _a(f["A"]))
    close(b.row_coordinates * s, f["A"]); close(b.col_coordinates * s, f["B"])
    close(b.row_contributions, f["rowfit"], 1e-8); close(b.col_contributions, f["colfit"], 1e-8)


# ----------------------------------------------------------------------------- bootstrap of PCoA and MDS
@pytest.mark.parametrize("method,seed", [("sampling", 21), ("permutation", 22)])
def test_bootstrap_pcoa_matches_multbiplotr(method, seed):
    f = F["bootpco"][method]
    b = bp.bootstrap_pcoa(np.array(F["D"]), nB=10, method=method, seed=seed, procrustes="MultBiplotR")
    close(b.eigenvalues, f["lambdas"], 1e-9)
    C = np.array([np.array(c) for c in (f["coords"].values() if isinstance(f["coords"], dict) else f["coords"])])
    close(np.array([c.T for c in b.coordinates]), C, 1e-9)
    Q = np.array([np.array(q) for q in (f["qual"].values() if isinstance(f["qual"], dict) else f["qual"])])
    close(np.array([q.T for q in b.qualities]), Q, 1e-8)


def test_orthogonal_procrustes_fits_better_than_multbiplotr_rotation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2)); th = 0.3
    Y = X @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    good = bp.simple_procrustes(X, Y); bad = bp.simple_procrustes(X, Y, "MultBiplotR")
    assert np.abs(good - X).max() < 1e-12 and np.abs(bad - X).max() > 0.1


def test_bootstrap_mds():
    f = F["bootmds"]
    b = bp.bootstrap_mds(np.array(F["D"]), nB=10, seed=23)
    s = _signs(b.reference, _a(f["conf0"]))
    close(b.reference * s, f["conf0"], 1e-9)
    close(np.transpose(np.array(b.coordinates), (1, 0, 2)) * s, np.array(f["conf"]), 1e-8)
    close(b.stress, f["stress"], 1e-10)


# ----------------------------------------------------------------------------- HE / multi-factor MANOVA
@pytest.mark.parametrize("ty", ["II", "III"])
def test_manova_terms_match_car(ty):
    h = F["he2"]; r = h[ty]
    m = bp.manova_terms(np.array(h["Y"]), pd.DataFrame({"fa": h["A"], "fb": h["B"]}), ["fa", "fb", "fa:fb"], ty)
    close(m["E"], r["SSPE"], 1e-10)
    assert m["dfe"] == r["dfe"]
    for t in ["fa", "fb", "fa:fb"]:
        close(m["terms"][t]["H"], r["SSP"][t], 1e-10)
        close(list(m["terms"][t]["stats"].values()), r["stats"][r["terms"].index(t)], 1e-10)
        assert m["terms"][t]["df"] == r["df"][r["terms"].index(t)]


def test_he_plot_terms_draws():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = F["he2"]
    m = bp.he_plot_terms(np.array(h["Y"]), pd.DataFrame({"fa": h["A"], "fb": h["B"]}))
    assert set(m["terms"]) == {"fa", "fb", "fa:fb"}
    plt.close("all")
