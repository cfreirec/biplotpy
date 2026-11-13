## Fixtures for phase 5 (biplotpy 0.4.0): constrained logistic biplots (binary and ordinal),
## external ordinal logistic biplot, PLSR1Bin, BinaryPLSR, log-frequency biplot,
## bootstrap of principal coordinates and of MDS, HE matrices of multi-factor MANOVA.
## MultBiplotR 25.11.15, smacof, car. Run from tests/fixtures: Rscript make_phase5.R
suppressPackageStartupMessages({ library(MultBiplotR); library(smacof); library(car); library(jsonlite) })
um <- function(x) unname(as.matrix(x))
out <- list()
set.seed(5)
n <- 60
Z <- matrix(rnorm(n * 2), n, 2)
X <- Z %*% matrix(c(1, 0.5, -0.8, 0.3, 0.2, 1, 0.6, -0.9), 2) + matrix(rnorm(n * 4, sd = 0.6), n, 4)
colnames(X) <- paste0("x", 1:4)
Bb <- matrix(c(1.5, -1, 0.5, 2, -1.2, 0.8, 0.7, 1.4, -1.6, 0.4, 1.1, -0.6), 6)
Yb <- matrix(as.numeric(runif(n * 6) < plogis(cbind(1, Z) %*% t(cbind(c(0, .5, -.5, .3, 0, -.2), Bb)))), n, 6)
colnames(Yb) <- paste0("b", 1:6)
Yo <- sapply(1:5, function(j) { lat <- Z %*% rnorm(2, sd = 1.3) + rlogis(n); as.integer(cut(lat, c(-Inf, -1.2, 0, 1.2, Inf))) })
colnames(Yo) <- paste0("o", 1:5)
Xb <- matrix(as.numeric(runif(n * 8) < plogis(Z %*% matrix(rnorm(16, sd = 1.5), 2))), n, 8)
colnames(Xb) <- paste0("xb", 1:8)
out$data <- list(X = um(X), Yb = um(Yb), Yo = um(Yo), Xb = um(Xb))
Yof <- as.data.frame(lapply(as.data.frame(Yo), function(v) factor(v, levels = 1:4, ordered = TRUE)))

## ---- constrained binary logistic biplot
cb <- ConstrainedLogisticBiplot(Yb, X, dim = 2, penalization = 0.1)
out$cbin <- list(Betas = um(cb$Betas), A = um(cb$RowCoordinates), B = um(cb$ColumnParameters),
  NullDev = unname(cb$NullDeviances), ModelDev = unname(cb$ModelDeviances), pvalues = c(cb$pvalues),
  CoxSnell = unname(cb$CoxSnell), Nagelkerke = unname(cb$Nagelkerke), MacFaden = unname(cb$MacFaden),
  Loadings = um(cb$Loadings), Communalities = unname(cb$Communalities), Sens = unname(cb$Sensitivity),
  Spec = unname(cb$Specificity), PercentsCorrec = unname(cb$PercentsCorrec), TotalPercent = cb$TotalPercent,
  sv_b0 = unname(cb$ContSupVarsBiplot$b0), sv_coords = um(cb$ContSupVarsBiplot$ColCoordinates),
  sv_R2 = unname(cb$ContSupVarsBiplot$R2))
## ---- constrained ordinal logistic biplot
co <- ConstrainedOrdinalLogisticBiplot(Yof, X, dim = 2, penalization = 0.1)
out$cord <- list(A = um(co$RowCoordinates), coef = um(co$ColumnParameters$coefficients),
  thr = um(co$ColumnParameters$thresholds), fit = um(co$ColumnParameters$fit), loadings = um(co$loadings),
  Communalities = c(co$Communalities), Deviance = co$Deviance, DevianceNull = co$DevianceNull,
  df = co$df, pval = co$pval, AIC = co$AIC, BIC = co$BIC)
## ---- external ordinal logistic biplot on the principal components of X
## AddOrdVars2Biplot converts the factors to integers and RidgeOrdinalLogistic then refuses them
## ("The dependent variable must be ordinal"); the reference is computed with that conversion off.
assignInNamespace("ConvertFactors2Integers", function(X) X, "MultBiplotR")
A0 <- prcomp(X, scale. = TRUE)$x[, 1:2]
eo <- AddOrdVars2Biplot(list(RowCoordinates = A0), Yof, penalization = 0.2, show = FALSE)$OrdSupVarsBiplot
out$eord <- list(A = um(A0), coef = um(eo$ColumnParameters$coefficients), thr = um(eo$ColumnParameters$thresholds),
  fit = um(eo$ColumnParameters$fit), loadings = um(eo$loadings), Deviance = eo$Deviance, pval = eo$pval)
## ---- PLSR1Bin (algorithm 1, Bastien et al.)
## In R >= 4.2 PLSR1Bin stops in RidgeBinaryLogistic (`if (class(y) == "formula")` on a matrix,
## whose class has length 2); the reference uses RidgeBinaryLogistic with class(y)[1].
rbl <- MultBiplotR::RidgeBinaryLogistic
body(rbl) <- parse(text = sub("clasey = class(y)", "clasey = class(y)[1]", paste(deparse(body(rbl)), collapse = "\n"), fixed = TRUE))[[1]]
assignInNamespace("RidgeBinaryLogistic", rbl, "MultBiplotR")
pb <- PLSR1Bin(Yb[, 1], X, S = 2)
out$plsr1bin <- list(T = um(pb$XScores), W = um(pb$XWeights), P = um(pb$XLoadings),
  beta = c(pb$BinaryFit$beta), Deviance = pb$BinaryFit$Deviance, structure = um(pb$XStructure))
## ---- BinaryPLSR (conjugate gradients of optim, seed 0)
invisible(capture.output(bp <- BinaryPLSR(Yb[, 1:3], Xb, S = 2, penalization = 0.1, seed = 0)))
out$binpls <- list(TT = um(bp$XScores), P = um(bp$XLoadings), U = um(bp$YScores), Q = um(bp$YLoadings),
  p0 = bp$InterceptsX, q0 = bp$InterceptsY, Expected = um(bp$Expected), PercentCorrect = bp$PercentCorrect)
## ---- log-frequency biplot (alternating least squares with random start, seed 11)
Nc <- matrix(rpois(15 * 6, lambda = exp(1 + Z[1:15, ] %*% matrix(c(.5, -.3, .2, .6, -.4, .1, .3, .5, -.6, .2, .4, -.2), 2))), 15, 6)
out$logfreq_N <- um(Nc)
set.seed(11)
invisible(capture.output(lf <- LogFrequencyBiplot(Nc)))
out$logfreq <- list(A = um(lf$RowCoordinates), B = um(lf$ColCoordinates), rowfit = um(lf$RowContributions),
  colfit = um(lf$ColContributions))
## ---- bootstrap of principal coordinates (residual resampling and permutation)
D <- as.matrix(dist(X[1:25, ])); out$D <- um(D)
set.seed(21); bs <- BootstrapDistance(D, nB = 10, method = "Sampling")
set.seed(22); bp2 <- BootstrapDistance(D, nB = 10, method = "Permutation")
pack <- function(b) list(lambdas = um(b$Eigenvalues), coords = lapply(b$Coordinates, um), qual = lapply(b$Qualities, um))
out$bootpco <- list(sampling = pack(bs), permutation = pack(bp2))
## ---- bootstrap of MDS (MultBiplotR::BootstrapSmacof fails: it reads Sol$Dhat, which SMACOF does
## not return). Reference: the same residual bootstrap on smacof::mds (ratio), refitted from the
## initial configuration and aligned by orthogonal Procrustes; negative resampled values are set to 0.
proc <- function(X, Y) { s <- svd(t(Y) %*% X); Y %*% (s$u %*% t(s$v)) }
m0 <- mds(D, ndim = 2, type = "ratio", init = "torgerson", itmax = 1000, eps = 1e-10)
Dn <- as.matrix(m0$dhat); Dh <- as.matrix(m0$confdist)
Er <- as.dist(Dn - Dh)
set.seed(23)
reps <- lapply(1:10, function(i) {
  DE <- as.matrix(sample(Er, length(Er), replace = TRUE) * as.dist(matrix(1, 25, 25))) + Dh
  DE[DE < 0] <- 0                                   # dissimilarities cannot be negative
  mi <- mds(DE, ndim = 2, type = "ratio", init = m0$conf, itmax = 1000, eps = 1e-10)
  list(conf = um(proc(m0$conf, mi$conf)), stress = mi$stress) })
out$bootmds <- list(conf0 = um(m0$conf), stress0 = m0$stress, conf = lapply(reps, `[[`, "conf"),
  stress = sapply(reps, `[[`, "stress"))
## ---- HE: two-factor MANOVA (unbalanced), car::Anova types II and III
set.seed(31)
fa <- factor(sample(c("a1", "a2", "a3"), 54, TRUE)); fb <- factor(sample(c("b1", "b2"), 54, TRUE))
Y3 <- cbind(y1 = rnorm(54) + (fa == "a2") + 0.5 * (fb == "b2"), y2 = rnorm(54) - 0.7 * (fa == "a3"),
            y3 = rnorm(54) + 0.6 * (fa == "a2") * (fb == "b2"))
out$he2 <- list(Y = um(Y3), A = as.character(fa), B = as.character(fb))
for (ty in c("II", "III")) {
  op <- options(contrasts = c("contr.sum", "contr.poly"))
  m <- lm(Y3 ~ fa * fb); av <- Anova(m, type = ty)
  options(op)
  terms <- names(av$SSP)
  st <- lapply(terms, function(t) { e <- Re(eigen(solve(av$SSPE) %*% av$SSP[[t]])$values)
    c(Pillai = sum(e / (1 + e)), Wilks = prod(1 / (1 + e)), HL = sum(e), Roy = max(e)) })
  out$he2[[ty]] <- list(terms = terms, SSP = lapply(av$SSP, um), SSPE = um(av$SSPE), df = unname(av$df),
                        dfe = av$error.df, stats = lapply(st, unname))
}
write_json(out, "phase5.json", digits = NA, auto_unbox = TRUE, matrix = "rowmajor")
cat("ok\n")
