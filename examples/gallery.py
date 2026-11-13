"""Gallery: renders every biplot of biplotpy into examples/gallery/*.png.

Run:  python examples/gallery.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import biplotpy as bp

OUT = Path(__file__).parent / "gallery"
OUT.mkdir(exist_ok=True)
FIX = Path(__file__).parents[1] / "tests" / "fixtures"
rng = np.random.default_rng(1)
iris = bp.load_iris()
X, sp = iris.iloc[:, :4], iris["Species"]


def save(name):
    plt.tight_layout(); plt.savefig(OUT / f"{name}.png", dpi=110); plt.close("all")


# 1 classical biplots ------------------------------------------------------------
hj = bp.hj_biplot(X, dim=3)
bp.plot_biplot(hj, groups=sp, group_shape="ellipse"); save("01_hj_ellipses")
jk = bp.jk_biplot(X)
bp.plot_biplot(jk, groups=sp, group_shape="hull"); save("02_jk_hulls")
gh = bp.gh_biplot(X)
bp.plot_biplot(gh, groups=sp, group_shape="bag", level=0.9); save("03_gh_alphabags")
bp.plot_biplot(jk, calibrated=True, show_rows=True); save("04_calibrated_axes")
bp.plot_biplot(jk, color_by="quality"); save("05_quality_colour")
bp.plot_biplot(jk, groups=sp, group_shape="density"); save("06_group_density")
bp.plot_correlation_circle(hj); save("07_correlation_circle")
bp.plot_scree(hj); save("08_scree")
bp.plot_contributions(hj, "cols", 0); save("09_contributions")
bp.plot_biplot_3d(hj, groups=sp); save("10_biplot_3d")
boot = bp.pca_bootstrap(X, dim=2, B=200, seed=1)
bp.plot_bootstrap(jk, boot); save("11_bootstrap")
# 2 canonical --------------------------------------------------------------------
cb = bp.canonical_biplot(X, sp)
bp.plot_canonical(cb); save("12_canonical")
bp.plot_canonical(cb, regions=True, arrows=False); save("13_cva_regions")
bp.plot_canonical(cb, calibrated=True, show_individuals=False); save("14_cva_calibrated")
bp.he_plot(X.to_numpy(), sp, variables=(2, 3), labels=("Petal.Length", "Petal.Width")); save("15_he_plot")
# 3 logistic ---------------------------------------------------------------------
n, p = 120, 8
A = rng.normal(size=(n, 2)); B = rng.normal(0, 1.5, (p, 2)); b0 = rng.normal(0, .5, p)
Xb = (rng.random((n, p)) < 1 / (1 + np.exp(-(b0 + A @ B.T)))).astype(float)
lbb = bp.binary_logistic_biplot(Xb, penalization=0.5)
bp.plot_logistic(lbb); save("16_binary_logistic")
Xo = np.zeros((n, 6))
for jj in range(6):
    eta = A @ B[jj]
    lat = eta + rng.logistic(size=n)
    Xo[:, jj] = np.digitize(lat, [-1.5, 0, 1.5]) + 1
lbo = bp.ordinal_logistic_biplot(Xo, penalization=0.3)
bp.plot_logistic(lbo); save("17_ordinal_logistic")
Xn = np.empty((n, 4), object)
for jj in range(4):
    e = np.column_stack([np.zeros(n), A @ rng.normal(0, 1.5, (2, 2)).T]); P = np.exp(e); P /= P.sum(1, keepdims=True)
    Xn[:, jj] = [["a", "b", "c"][rng.choice(3, p=q)] for q in P]
lbn = bp.nominal_logistic_biplot(Xn, penalization=0.3)
bp.plot_nominal_regions(lbn, 0); save("18_nominal_regions")
# 4 STATIS family, dynamic, unfolding, DISTATIS, CDPCA ------------------------------
F2 = json.loads((FIX / "phase2.json").read_text())
tabs = {k: np.array(v) for k, v in F2["statis"]["X"].items()}
st = bp.statis(tabs)
bp.plot_statis(st, "interstructure"); save("19_statis_interstructure")
bp.plot_statis(st, "compromise", trajectories=True, cols=False); save("20_statis_trajectories")
cs = bp.canonical_statis({k: np.array(v) for k, v in F2["canstatis"]["X"].items()}, F2["canstatis"]["groups"])
bp.plot_statis(cs, "compromise", trajectories=True); save("21_canonical_statis")
occ = [X.to_numpy()[::10] + 0.25 * t + rng.normal(0, .1, (15, 4)) for t in range(4)]
dy = bp.dynamic_biplot(occ, alpha=2)
bp.plot_dynamic(dy); save("22_dynamic")
uf = bp.unfolding(np.array(F2["unfolding"]["delta"]))
bp.plot_unfolding(uf); save("23_unfolding")
F3 = json.loads((FIX / "phase3.json").read_text())
ds = bp.distatis(np.stack([np.array(c) for c in F3["distatis"]["cube"]], 2))
bp.plot_distatis(ds); save("24_distatis")
cd = bp.cdpca(np.array(F3["cdpca"]["X"]), 3, 2, nstart=10, seed=1)
bp.plot_cdpca(cd); save("25_cdpca")
# 5 GGE and AMMI -----------------------------------------------------------------
gdat = pd.DataFrame(np.array(F2["gge"]["data"]), index=[f"G{i + 1}" for i in range(18)],
                    columns=["BH93", "EA93", "HW93", "ID93", "KE93", "NN93", "OA93", "RN93", "WP93"])
gm = bp.gge_model(gdat, centering="tester", scaling="none", svp="row")
fig, axs = plt.subplots(2, 5, figsize=(26, 11))
for ax, t in zip(axs.ravel(), ["biplot", "examine_env", "examine_gen", "env_relationship", "compare_gens",
                               "which_won_where", "discrimination", "rank_env", "mean_stability", "rank_gen"]):
    bp.plot_gge(gm if t not in ("discrimination", "rank_env", "env_relationship", "examine_gen") else
                bp.gge_model(gdat, svp="column"), t, ax=ax, env="BH93", gen="G1", gen2="G2")
save("26_gge_views")
am = bp.ammi(F3 and F2["ammi"]["env"], F2["ammi"]["gen"], F2["ammi"]["rep"], F2["ammi"]["y"])
fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
bp.plot_ammi(am, "AMMI1", ax=axs[0]); bp.plot_ammi(am, "AMMI2", ax=axs[1]); save("27_ammi")
# 6 correspondence-type ----------------------------------------------------------
F4 = json.loads((FIX / "phase4.json").read_text())
smoke = pd.DataFrame(np.array(F4["ca"]["N"]), index=["SM", "JM", "SE", "JE", "SC"], columns=["none", "light", "medium", "heavy"])
bp.plot_ordination(bp.ca(smoke)); save("28_ca")
mc = bp.mca(pd.DataFrame(F4["mca"]["X"]))
bp.plot_ordination(mc, show_rows=True, col_labels=True); save("29_mca")
Y, Z = np.array(F4["cca"]["Y"]), pd.DataFrame(np.array(F4["cca"]["Z"]), columns=["N", "P", "K", "Ca", "Al"])
bp.plot_ordination(bp.cca(Y, Z), col_labels=False); save("30_cca_triplot")
bp.plot_ordination(bp.rda(Y, Z, scaling=2), col_labels=False); save("31_rda_triplot")
# 7 distances --------------------------------------------------------------------
D = bp.continuous_distances(X.to_numpy())
pc = bp.pcoa(D)
sv = bp.add_continuous_variables(pc, X)
bp.plot_ordination(pc, groups=sp, sup_vars=sv); save("32_pcoa")
md = bp.mds(D[:60, :60], type="ordinal")
bp.plot_ordination(md, groups=sp[:60]); save("33_mds_ordinal")
Bin = np.array(F4["pcoa"]["Bin"])
bpco = bp.pcoa(bp.binary_distances(Bin, coefficient="Jaccard"))
ext = bp.external_logistic_biplot(bpco.row_coordinates, Bin, 0.2, col_names=[f"b{k + 1}" for k in range(8)])
bp.plot_ordination(ext); save("34_external_logistic")
cda = bp.canonical_distance_analysis(D, sp, X=X.to_numpy(), nperm=199, seed=1)
bp.plot_ordination(cda); save("35_canonical_distances")
# 8 factor-type ------------------------------------------------------------------
fa = bp.fa_biplot(X, 2, "varimax")
bp.plot_biplot(fa, groups=sp); save("36_fa_biplot")
Xm = X.to_numpy().copy(); Xm[rng.random(Xm.shape) < 0.1] = np.nan
nb = bp.nipals_biplot(Xm, 2, alpha=1)
bp.plot_biplot(nb, groups=sp); save("37_nipals_missing")
oo = F4["pls"]
pb = bp.pls_biplot(pd.DataFrame(np.array(oo["X"]), columns=["Acidity", "Peroxide", "K232", "K270", "DK"]),
                   pd.DataFrame(np.array(oo["Y"]), columns=["yellow", "green", "brown", "glossy", "transp", "syrup"]))
bp.plot_biplot(pb, row_labels=True); save("38_pls_biplot")
co = bp.coinertia(np.array(F4["coin"]["X"]), np.array(F4["coin"]["Y"]), True, False)
bp.plot_coinertia(co); save("39_coinertia")
pbp = bp.polychoric_biplot(np.array(F4["poly"]["X"]).astype(int)[:150])
bp.plot_logistic(pbp); save("40_polychoric_biplot")
# 9 phase 5 (0.4.0) ---------------------------------------------------------------
F5 = json.loads((FIX / "phase5.json").read_text())
d5 = F5["data"]
X5, Yb5, Yo5, Xb5 = (np.array(d5[k], float) for k in ("X", "Yb", "Yo", "Xb"))
bp.constrained_logistic_biplot(Yb5, X5, penalization=0.1).plot(); save("41_constrained_binary_logistic")
bp.constrained_ordinal_logistic_biplot(Yo5, X5, penalization=0.1).plot(); save("42_constrained_ordinal_logistic")
pco5 = bp.pcoa(bp.continuous_distances(X5))
bp.external_ordinal_logistic_biplot(pco5, Yo5).plot(); save("43_external_ordinal_logistic")
bp.plot_plsbin(bp.plsr1_bin(Yb5[:, 0], pd.DataFrame(X5, columns=[f"x{k + 1}" for k in range(4)]))); save("44_plsr1_bin")
bp.plot_plsbin(bp.binary_plsr(pd.DataFrame(Yb5[:, :3], columns=["b1", "b2", "b3"]),
                              pd.DataFrame(Xb5, columns=[f"xb{k + 1}" for k in range(8)]), seed=0)); save("45_binary_plsr")
bp.log_frequency_biplot(np.array(F5["logfreq_N"]), seed=11).plot(); save("46_log_frequency")
bp.bootstrap_pcoa(np.array(F5["D"]), nB=200, seed=1).plot(); save("47_bootstrap_pcoa")
bp.bootstrap_mds(np.array(F5["D"]), nB=100, seed=1).plot(kind="hull"); save("48_bootstrap_mds")
h5 = F5["he2"]
bp.he_plot_terms(np.array(h5["Y"]), pd.DataFrame({"A": h5["A"], "B": h5["B"]}), labels=("y1", "y2")); save("49_he_two_factors")
print("rendered", len(list(OUT.glob("*.png"))), "figures")
