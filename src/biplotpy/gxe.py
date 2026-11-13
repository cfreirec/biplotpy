"""Genotype x environment biplots: GGE biplot (Yan et al., 2000; port of
``GGEBiplots::GGEModel``) and AMMI (Gauch, 1992; port of ``agricolae::AMMI``)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def gge_model(data, centering="tester", scaling="none", svp="column") -> dict:
    """GGE biplot of a genotype (rows) x environment (columns) table of means.

    ``centering``: none, global, tester (environment-centred, default) or
    double; ``scaling``: none or sd; ``svp``: row, column, dual, symmetrical or
    gge. Coordinates are rescaled as in GGEBiplots for display."""
    if isinstance(data, pd.DataFrame):
        gen, env = [str(i) for i in data.index], [str(c) for c in data.columns]
        D = data.to_numpy(float)
    else:
        D = np.asarray(data, float)
        gen = [f"G{i + 1}" for i in range(D.shape[0])]
        env = [f"E{j + 1}" for j in range(D.shape[1])]
    if np.isnan(D).any():
        raise ValueError("missing data in input")
    if centering in (1, "global"):
        D = D - D.mean()
    elif centering in (2, "tester"):
        D = D - D.mean(0)
    elif centering in (3, "double"):
        D = D + D.mean() - D.mean(0)[None, :] - D.mean(1)[:, None]
    if scaling in (1, "sd"):
        D = D / D.std(0, ddof=1)
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    V = Vt.T
    n = D.shape[0]

    def ratio(a, b):
        return max((a[:, 0].max() - a[:, 0].min()) / (b[:, 0].max() - b[:, 0].min()),
                   (a[:, 1].max() - a[:, 1].min()) / (b[:, 1].max() - b[:, 1].min()))
    if svp in (0, "gge"):
        G = U * np.sqrt(n - 1)
        E = D.T @ U / np.sqrt(n - 1)
        E = E / ratio(E, G)
    elif svp in (1, "row"):
        G = U * s
        E = V / ratio(V, G)
    elif svp in (2, "column"):
        E = V * s
        G = U / ratio(U, E)
    elif svp in (3, "dual"):
        G, E = U * s, V * s
    elif svp in (4, "symmetrical"):
        G, E = U * np.sqrt(s), V * np.sqrt(s)
    else:
        raise ValueError(f"unknown SVP {svp}")
    vartotal = np.sum(s ** 2)
    return {"coordgenotype": G, "coordenviroment": E, "eigenvalues": s, "vartotal": vartotal,
            "varexpl": s ** 2 / vartotal * 100, "labelgen": gen, "labelenv": env, "data": D,
            "centering": centering, "scaling": scaling, "svp": svp}


def ammi(env, gen, rep, y) -> dict:
    """AMMI analysis from plot data (environment, genotype, replicate within
    environment, response). Returns the ANOVA, the decomposition of the
    interaction in principal components (Gollob degrees of freedom) and the
    biplot scores, as ``agricolae::AMMI`` with replicates."""
    df = pd.DataFrame({"ENV": pd.Categorical(np.asarray(env).astype(str)),
                       "GEN": pd.Categorical(np.asarray(gen).astype(str)),
                       "REP": pd.Categorical(np.asarray(rep).astype(str)), "Y": np.asarray(y, float)})
    e_lev = sorted(df["ENV"].unique()); g_lev = sorted(df["GEN"].unique())
    nenv, ngen = len(e_lev), len(g_lev)
    nrep = df["REP"].nunique()
    gm = df["Y"].mean()
    # sequential sums of squares for Y ~ ENV + REP%in%ENV + GEN + ENV:GEN (balanced design)
    me = df.groupby("ENV", observed=True)["Y"].mean()
    mg = df.groupby("GEN", observed=True)["Y"].mean()
    mre = df.groupby(["ENV", "REP"], observed=True)["Y"].mean()
    mge = df.groupby(["ENV", "GEN"], observed=True)["Y"].mean()
    N = len(df)
    ss_env = N / nenv * np.sum((me - gm) ** 2)
    ss_rep = np.sum(df.groupby(["ENV", "REP"], observed=True).size() * (mre.to_numpy() - me.reindex(mre.index.get_level_values(0)).to_numpy()) ** 2)
    ss_gen = N / ngen * np.sum((mg - gm) ** 2)
    cell = mge.reset_index()
    cell["int"] = (cell["Y"].to_numpy() - me.reindex(cell["ENV"].astype(str)).to_numpy()
                   - mg.reindex(cell["GEN"].astype(str)).to_numpy() + gm)
    ss_ge = nrep * np.sum(cell["int"] ** 2)
    ss_tot = np.sum((df["Y"] - gm) ** 2)
    ss_res = ss_tot - ss_env - ss_rep - ss_gen - ss_ge
    df_env, df_rep, df_gen = nenv - 1, nenv * (nrep - 1), ngen - 1
    df_ge = df_env * df_gen
    df_res = N - 1 - df_env - df_rep - df_gen - df_ge
    mse = ss_res / df_res
    rows = {"ENV": (df_env, ss_env), "REP(ENV)": (df_rep, ss_rep), "GEN": (df_gen, ss_gen),
            "ENV:GEN": (df_ge, ss_ge), "Residuals": (df_res, ss_res)}
    anova = pd.DataFrame(rows, index=["Df", "Sum Sq"]).T
    anova["Mean Sq"] = anova["Sum Sq"] / anova["Df"]
    anova["F value"] = np.nan; anova["Pr(>F)"] = np.nan
    ms = anova["Mean Sq"]
    anova.loc["ENV", "F value"] = ms["ENV"] / ms["REP(ENV)"]
    anova.loc["ENV", "Pr(>F)"] = sps.f.sf(anova.loc["ENV", "F value"], df_env, df_rep)
    for r in ("REP(ENV)", "GEN", "ENV:GEN"):
        anova.loc[r, "F value"] = ms[r] / mse
        anova.loc[r, "Pr(>F)"] = sps.f.sf(anova.loc[r, "F value"], anova.loc[r, "Df"], df_res)
    # interaction matrix (genotypes x environments) and its SVD
    I = cell.pivot(index="GEN", columns="ENV", values="int").loc[g_lev, e_lev].to_numpy()
    U, L, Vt = np.linalg.svd(I, full_matrices=False)
    minimo = min(ngen, nenv)
    L = L[:minimo]
    SS = L ** 2 * nrep
    percent = SS / SS.sum() * 100
    out = []
    for i in range(1, minimo):
        DF = (ngen - 1) + (nenv - 1) - (2 * i - 1)
        if DF <= 0:
            break
        msi = SS[i - 1] / DF
        F = msi / mse
        out.append([percent[i - 1], percent[:i].sum(), DF, SS[i - 1], msi, F, sps.f.sf(F, DF, df_res)])
    ssammi = pd.DataFrame(out, columns=["percent", "acum", "Df", "Sum Sq", "Mean Sq", "F value", "Pr.F"],
                          index=[f"PC{i + 1}" for i in range(len(out))])
    LL = np.sqrt(L)
    scores_g = U * LL
    scores_e = Vt.T * LL
    means_g = mge.reset_index().groupby("GEN", observed=True)["Y"].mean().loc[g_lev]
    means_e = me.loc[e_lev]
    return {"anova": anova, "analysis": ssammi, "genotype_scores": pd.DataFrame(scores_g, index=g_lev),
            "environment_scores": pd.DataFrame(scores_e, index=e_lev), "genotype_means": means_g,
            "environment_means": means_e, "interaction": pd.DataFrame(I, index=g_lev, columns=e_lev), "MSE": mse}
