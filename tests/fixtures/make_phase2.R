## Fixtures for biplotpy phase 2: STATIS family, unfolding, GGE, AMMI, polr, multinom,
## binary logistic objective. Output: biplotpy/tests/fixtures/phase2.json
suppressPackageStartupMessages({
  library(MultBiplotR); library(smacof); library(GGEBiplots); library(agricolae)
  library(MASS); library(nnet); library(ade4); library(jsonlite)
})
set.seed(20261023)
out <- list()
um <- function(m) unname(as.matrix(m))

## ---- STATIS (same rows) ----
n <- 12
Ls <- list(T1 = matrix(rnorm(n * 4), n, 4), T2 = matrix(rnorm(n * 3), n, 3), T3 = matrix(rnorm(n * 5), n, 5))
Ls$T2[, 1] <- Ls$T1[, 1] + 0.3 * Ls$T2[, 1]; Ls$T3[, 2] <- Ls$T1[, 2] - 0.5 * Ls$T3[, 2]
for (k in names(Ls)) { rownames(Ls[[k]]) <- paste0("r", 1:n); colnames(Ls[[k]]) <- paste0(k, "v", 1:ncol(Ls[[k]])) }
st <- StatisBiplot(Ls, dimens = 2)
out$statis <- list(X = lapply(Ls, um), RV = um(st$RV), weights = st$Weights, eig = st$Biplot$EigenValues,
                   rows = um(st$Biplot$RowCoordinates), cols = um(st$Biplot$ColCoordinates),
                   scf = st$Biplot$Scale_Factor, traj1 = um(st$TrajInd[[1]]), eiginter = unname(st$EigInter))
## RV from ade4 (independent implementation), tables standardized beforehand
kt <- ktab.list.df(lapply(Ls, function(x) as.data.frame(scale(x))))
out$statis$RV_ade4 <- um(statis(kt, scannf = FALSE)$RV)

## ---- Canonical STATIS ----
n2 <- 45; grp <- factor(rep(c("A", "B", "C"), each = 15))
mk <- function(p, shift) { x <- matrix(rnorm(n2 * p), n2, p); x[grp == "B", 1] <- x[grp == "B", 1] + shift; x[grp == "C", 2] <- x[grp == "C", 2] + shift; x }
Lc <- list(O1 = mk(4, 1.5), O2 = mk(4, 1.0), O3 = mk(3, 2.0))
for (k in names(Lc)) colnames(Lc[[k]]) <- paste0(k, "v", 1:ncol(Lc[[k]]))
invisible(capture.output(cs <- CanonicalStatisBiplot(Lc, grp, dimens = 2)))
out$canstatis <- list(X = lapply(Lc, um), groups = as.character(grp), RV = um(cs$RV), weights = cs$Weights,
                      eig = cs$Biplot$EigenValues, rows = um(cs$Biplot$RowCoordinates),
                      cols = um(cs$Biplot$ColCoordinates), scf = cs$Biplot$Scale_Factor)

## ---- Dual STATIS: MultBiplotR::DualStatisBiplot is unfinished; reproduce its core lines ----
Ld <- list(D1 = matrix(rnorm(10 * 4), 10, 4), D2 = matrix(rnorm(14 * 4), 14, 4), D3 = matrix(rnorm(8 * 4), 8, 4))
Ld$D2[, 2] <- Ld$D2[, 1] + 0.4 * Ld$D2[, 2]
for (k in names(Ld)) colnames(Ld[[k]]) <- paste0("v", 1:4)
err <- tryCatch({ MultBiplotR:::DualStatisBiplot(Ld); "no error" }, error = function(e) conditionMessage(e))
Xd <- MultBiplotR:::MultiTableTransform(Ld, InitTransform = "Standardize columns", dual = TRUE)
Wt <- lapply(Xd$X, function(x) t(x) %*% x)
P <- outer(1:3, 1:3, Vectorize(function(i, j) sum(diag(Wt[[i]] %*% t(Wt[[j]])))))
RV <- diag(1 / sqrt(diag(P))) %*% P %*% diag(1 / sqrt(diag(P)))
w <- abs(svd(RV)$u[, 1]); w <- w / sum(w)
W <- Reduce(`+`, Map(`*`, w, Wt)); sv <- svd(W)
out$dualstatis <- list(X = lapply(Ld, um), error = err, sd = Xd$SD, RV = RV, weights = w, eig = sv$d,
                       cols = sv$u[, 1:2] %*% diag(sqrt(sv$d[1:2])))

## ---- Unfolding (ratio) ----
Dl <- um(as.matrix(breakfast))[1:20, ]
uf <- unfolding(Dl, type = "ratio", ndim = 2)
out$unfolding <- list(delta = Dl, stress = uf$stress, conf_row = um(uf$conf.row), conf_col = um(uf$conf.col),
                      niter = uf$niter, confdist = um(uf$confdist))

## ---- GGE ----
data(Ontario, package = "GGEBiplots"); yan.winterwheat <- Ontario
gg <- list(data = um(yan.winterwheat))
for (svp in c("row", "column", "dual", "symmetrical")) for (cen in c("tester", "double", "global")) {
  m <- GGEModel(yan.winterwheat, centering = cen, scaling = "sd", SVP = svp)
  gg[[paste(svp, cen, sep = "_")]] <- list(G = um(m$coordgenotype), E = um(m$coordenviroment), d = m$eigenvalues,
                                          varexpl = m$varexpl)
}
out$gge <- gg

## ---- AMMI ----
data(plrv, package = "agricolae")
am <- with(plrv, AMMI(Locality, Genotype, Rep, Yield, console = FALSE))
out$ammi <- list(env = as.character(plrv$Locality), gen = as.character(plrv$Genotype), rep = as.character(plrv$Rep),
                 y = plrv$Yield, anova = um(am$ANOVA[, 1:4]), anova_rows = rownames(am$ANOVA),
                 analysis = um(am$analysis[, c("percent", "acum", "Df", "Sum.Sq", "Mean.Sq", "F.value", "Pr.F")]),
                 biplot = am$biplot)

## ---- Ordinal (polr) and nominal (multinom) with fixed row scores ----
na <- 200; A <- matrix(rnorm(na * 2), na, 2)
eta <- A %*% c(1.2, -0.7); lat <- eta + rlogis(na)
yo <- cut(lat, c(-Inf, -1, 0.3, 1.5, Inf), labels = FALSE)
po <- polr(factor(yo) ~ A, method = "logistic")
yn <- sapply(1:na, function(i) { e <- c(0, 0.5 + A[i, ] %*% c(1, -1), -0.3 + A[i, ] %*% c(-0.8, 0.6)); sample(1:3, 1, prob = exp(e) / sum(exp(e))) })
mn <- multinom(factor(yn) ~ A, trace = FALSE, reltol = 1e-12, maxit = 1000)
out$polr <- list(A = A, y = yo, zeta = unname(po$zeta), coef = unname(po$coefficients), deviance = po$deviance)
out$multinom <- list(y = yn, coef = um(coef(mn)), deviance = deviance(mn))

## ---- Binary logistic biplot: correctly defined objective, optimized in R ----
nb <- 40; pb <- 8
At <- matrix(rnorm(nb * 2), nb, 2); Bt <- cbind(rnorm(pb, 0, 0.5), matrix(rnorm(pb * 2, 0, 1.5), pb, 2))
Xb <- matrix(rbinom(nb * pb, 1, plogis(cbind(1, At) %*% t(Bt))), nb, pb)
lam <- 0.5
obj <- function(par) {
  A <- matrix(par[1:(nb * 2)], nb, 2); B <- matrix(par[-(1:(nb * 2))], pb, 3)
  e <- cbind(1, A) %*% t(B)
  -sum(Xb * plogis(e, log.p = TRUE) + (1 - Xb) * plogis(-e, log.p = TRUE)) + lam / 2 * (sum(A^2) + sum(B[, 2:3]^2))
}
gr <- function(par) {
  A <- matrix(par[1:(nb * 2)], nb, 2); B <- matrix(par[-(1:(nb * 2))], pb, 3)
  E <- plogis(cbind(1, A) %*% t(B)) - Xb
  c(E %*% B[, 2:3] + lam * A, cbind(colSums(E), t(E) %*% A + lam * B[, 2:3]))
}
Xs <- scale(Xb); Xs[is.nan(Xs)] <- 0; s <- svd(Xs)
A0 <- s$u[, 1:2] %*% diag(s$d[1:2]) / sqrt(nb)
op <- optim(c(A0, rep(0, pb * 3)), obj, gr, method = "BFGS", control = list(maxit = 20000, reltol = 1e-14))
Af <- matrix(op$par[1:(nb * 2)], nb, 2); Bf <- matrix(op$par[-(1:(nb * 2))], pb, 3)
out$binary <- list(X = Xb, lambda = lam, value = op$value, eta = cbind(1, Af) %*% t(Bf), A0 = A0,
                   convergence = op$convergence)

write_json(out, "/home/claude/py/biplotpy/tests/fixtures/phase2.json", digits = NA, auto_unbox = TRUE, matrix = "rowmajor")
cat("ok\n")
