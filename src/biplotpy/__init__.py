"""biplotpy: biplot methods of the Salamanca school for Python.

Classical biplots (GH, JK, SQRT and HJ-Biplot), canonical/MANOVA biplot with
confidence circles and MANOVA tests, bootstrap of the PCA biplot,
supplementary variables and calibrated axes; logistic biplots (binary,
ordinal, nominal), STATIS-ACT, dual and canonical STATIS, dynamic biplot,
metric unfolding, GGE and AMMI; DISTATIS and CDPCA; Mardia's test,
heteroscedastic MANOVA (WTS/MATS with bootstrap) and HE plots (one or several
factors); constrained logistic biplots (binary and ordinal), external ordinal
logistic biplot, PLS regression with binary responses (PLSR1Bin, BinaryPLSR),
log-frequency biplot, bootstrap of principal coordinates and of MDS. Ported from the R package
MultBiplotR (Vicente-Villardón, Vicente-González and Frutos-Bernal) and
validated numerically against it.
"""
from .transforms import initial_transform, TRANSFORMS, FIXES
from .classical import ContinuousBiplot, pca_biplot, hj_biplot, gh_biplot, jk_biplot, sqrt_biplot
from .canonical import CanonicalBiplot, canonical_biplot, canonical_variate_analysis
from .extras import pca_bootstrap, add_continuous_variables, project_rows, biplot_scales, calibrated_axis
from .utils import r_pretty, matrix_sqrt
from .datasets import load_iris, load_mtcars
from .logistic import (LogisticBiplot, binary_logistic_biplot, ordinal_logistic_biplot, ordinal_logistic_fit,
                       nominal_logistic_biplot, nominal_logistic_fit)
from .statis import StatisResult, statis, dual_statis, canonical_statis
from .dynamic import dynamic_biplot
from .unfolding import unfolding
from .gxe import gge_model, ammi
from .multitest import (mardia, manova_wts, he_matrices, he_ellipses, he_plot, roy_crit, hlt_crit, manova_terms,
                        he_plot_terms)
from .restricted import (constrained_logistic_biplot, constrained_ordinal_logistic_biplot,
                         external_ordinal_logistic_biplot, ridge_ordinal_logistic)
from .plsbin import plsr1_bin, binary_plsr
from .logfreq import criss_cross, log_frequency_biplot
from .bootdist import OrdinationBootstrap, bootstrap_pcoa, bootstrap_mds, simple_procrustes
from .rrng import RRNG
from .roptim import optim_cg
from .distatis import distatis
from .cdpca import cdpca
from .correspondence import OrdinationBiplot, ca, mca, cca, rda, indicator_matrix
from .distances import (binary_distances, binary_similarity, continuous_distances, gower_distances, pcoa, mds,
                        ridge_logistic, external_logistic_biplot, canonical_distance_analysis)
from .factor import (rotate, fa_biplot, nipals, nipals_biplot, pls_regression, pls_biplot, coinertia, polychoric,
                     polychoric_biplot)
from .plot import (plot_biplot, plot_canonical, plot_correlation_circle, plot_scree, plot_contributions,
                   plot_biplot_3d, plot_bootstrap, plot_logistic, plot_nominal_regions, plot_statis, plot_dynamic,
                   plot_unfolding, plot_distatis, plot_cdpca, plot_gge, plot_ammi, plot_ordination, plot_coinertia,
                   concentration_ellipse, convex_hull, alpha_bag, plot_bootstrap_ordination, plot_plsbin)

__version__ = "0.4.0"
__all__ = ["initial_transform", "TRANSFORMS", "FIXES", "ContinuousBiplot", "pca_biplot", "hj_biplot",
           "gh_biplot", "jk_biplot", "sqrt_biplot", "CanonicalBiplot", "canonical_biplot",
           "canonical_variate_analysis", "pca_bootstrap", "add_continuous_variables", "project_rows",
           "biplot_scales", "calibrated_axis", "r_pretty", "matrix_sqrt", "load_iris", "load_mtcars",
           "LogisticBiplot", "binary_logistic_biplot", "ordinal_logistic_biplot", "ordinal_logistic_fit",
           "nominal_logistic_biplot", "nominal_logistic_fit", "StatisResult", "statis", "dual_statis",
           "canonical_statis", "dynamic_biplot", "unfolding", "gge_model", "ammi",
           "mardia", "manova_wts", "he_matrices", "he_ellipses", "he_plot", "roy_crit", "hlt_crit",
           "distatis", "cdpca",
           "OrdinationBiplot", "ca", "mca", "cca", "rda", "indicator_matrix", "binary_distances",
           "binary_similarity", "continuous_distances", "gower_distances", "pcoa", "mds", "ridge_logistic",
           "external_logistic_biplot", "canonical_distance_analysis", "rotate", "fa_biplot", "nipals",
           "nipals_biplot", "pls_regression", "pls_biplot", "coinertia", "polychoric", "polychoric_biplot",
           "plot_biplot", "plot_canonical", "plot_correlation_circle", "plot_scree", "plot_contributions",
           "plot_biplot_3d", "plot_bootstrap", "plot_logistic", "plot_nominal_regions", "plot_statis",
           "plot_dynamic", "plot_unfolding", "plot_distatis", "plot_cdpca", "plot_gge", "plot_ammi",
           "plot_ordination", "plot_coinertia", "concentration_ellipse", "convex_hull", "alpha_bag",
           "manova_terms", "he_plot_terms", "constrained_logistic_biplot", "constrained_ordinal_logistic_biplot",
           "external_ordinal_logistic_biplot", "ridge_ordinal_logistic", "plsr1_bin", "binary_plsr", "criss_cross",
           "log_frequency_biplot", "OrdinationBootstrap", "bootstrap_pcoa", "bootstrap_mds", "simple_procrustes",
           "RRNG", "optim_cg", "plot_bootstrap_ordination", "plot_plsbin"]
