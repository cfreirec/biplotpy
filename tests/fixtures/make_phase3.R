## Fixtures for phase 3: Mardia (psych), MANOVA.RM statistics, HE matrices (car),
## DISTATIS (DistatisR core), CDPCA (biplotbootGUI::CDpca from a given start),
## multivariate meta-analysis (mixmeta). Sources of DistatisR, MANOVA.RM and
## biplotbootGUI are taken from the CRAN mirrors on GitHub (cloned in ../gh).
suppressPackageStartupMessages({ library(psych); library(car); library(mixmeta); library(MASS); library(jsonlite) })
gh <- "/home/claude/gh"
set.seed(20261024)
out <- list(); um <- function(m) unname(as.matrix(m))

## ---- Mardia ----
Xm <- matrix(rexp(120 * 3), 120, 3)
mi <- mardia(iris[, 1:4], plot = FALSE); me <- mardia(Xm, plot = FALSE)
pick <- function(m) m[c("b1p", "b2p", "skew", "small.skew", "p.skew", "p.small", "kurtosis", "p.kurt")]
out$mardia <- list(iris = pick(mi), exp = c(list(X = Xm), pick(me)))

## ---- MANOVA.RM (statistics; magic::adiag replaced by an equivalent block diagonal) ----
adiag_ <- function(a, b) { a <- as.matrix(a); b <- as.matrix(b); m <- matrix(0, nrow(a) + nrow(b), ncol(a) + ncol(b))
  m[seq_len(nrow(a)), seq_len(ncol(a))] <- a; m[nrow(a) + seq_len(nrow(b)), ncol(a) + seq_len(ncol(b))] <- b; m }
src <- readLines(file.path(gh, "MANOVA.RM/R/MANOVA_Statistics.R"))
src <- gsub("magic::adiag", "adiag_", src, fixed = TRUE); eval(parse(text = src))
g3 <- factor(rep(c("a", "b", "c"), c(12, 20, 9)))
Ym <- rbind(mvrnorm(12, c(0, 0, 0), diag(3)), mvrnorm(20, c(0.5, 0, 0), diag(c(4, 1, 2))), mvrnorm(9, c(0, 0.8, 0), diag(3) * 0.5))
p <- 3; a <- 3; H <- (diag(a) - matrix(1 / a, a, a)) %x% diag(p)
x <- as.vector(t(Ym))
rp <- MANOVA.Stat(x, n = as.vector(table(g3)), hypo_matrix = H, iter = 20000, alpha = 0.05, resampling = "paramBS",
                  n.groups = a, p = p, para = FALSE, CPU = 1, seed = 1)
rw <- MANOVA.Stat(x, n = as.vector(table(g3)), hypo_matrix = H, iter = 20000, alpha = 0.05, resampling = "WildBS",
                  n.groups = a, p = p, para = FALSE, CPU = 1, seed = 1)
out$manova <- list(Y = Ym, g = as.character(g3), WTS = rp$WTS, MATS = as.numeric(rp$MATS), p_param = rp$WTPS,
                   p_wild = rw$WTPS)

## ---- HE matrices (car::Anova) and critical values ----
mod <- lm(cbind(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width) ~ Species, data = iris)
av <- Anova(mod)
roy <- function(alpha, p, dfh, dfe) { df1 <- max(p, dfh); df2 <- dfe - df1 + dfh; (df1 / df2) * qf(alpha, df1, df2, lower.tail = FALSE) }
out$he <- list(H = um(av$SSP$Species), E = um(av$SSPE), dfh = 2, dfe = 147, roy = roy(0.05, 4, 2, 147),
               stats = sapply(c("Pillai", "Wilks", "Hotelling-Lawley", "Roy"), function(t) summary(av, test = t)$multivariate.tests$Species$test))
out$he$stats <- sapply(c("Pillai", "Wilks", "Hotelling-Lawley", "Roy"), function(t) {
  e <- Re(eigen(solve(av$SSPE) %*% av$SSP$Species)$values)
  switch(t, Pillai = sum(e / (1 + e)), Wilks = prod(1 / (1 + e)), `Hotelling-Lawley` = sum(e), Roy = max(e)) })

## ---- DISTATIS ----
for (f in c("DblCenterDist", "Dist2CP", "MFAnormCP", "CP2MFAnormedCP", "SUMPCAnormCP", "CP2SUMPCAnormedCP",
            "NuclearnormCP", "CP2NuclearNormedCP", "GetCmat", "ComputeSplus", "rldiag", "distatis"))
  source(file.path(gh, "DistatisR/R", paste0(f, ".R")))
base <- matrix(rnorm(10 * 3), 10, 3)
cube <- array(0, c(10, 10, 4))
for (t in 1:4) cube[, , t] <- as.matrix(dist(base + matrix(rnorm(30, 0, 0.2 * t), 10, 3)))^2
out$distatis <- list(cube = lapply(1:4, function(t) cube[, , t]))
for (nm in c("MFA", "SUMPCA", "NUCLEAR")) {
  d <- distatis(cube, Norm = nm, nfact2keep = 3)
  out$distatis[[nm]] <- list(C = um(d$res4Cmat$C), alpha = unname(d$res4Cmat$alpha), Splus = um(d$res4Splus$Splus),
                             eig = unname(d$res4Splus$eigValues), F = um(d$res4Splus$F),
                             partial1 = um(d$res4Splus$PartialF[, , 1]), G = um(d$res4Cmat$G))
}

## ---- CDPCA from a fixed start ----
txt <- paste(readLines(file.path(gh, "biplotbootGUI/R/CDpca.R")), collapse = "\n")
txt <- sub("CDpca <- function(data, class = NULL,", "CDpca0 <- function(data, U0, V0, class = NULL,", txt, fixed = TRUE)
txt <- sub("U <- RandMat(I, P)", "U <- U0", txt, fixed = TRUE); txt <- sub("V <- RandMat(J, Q)", "V <- V0", txt, fixed = TRUE)
eval(parse(text = txt))
powerMethod <- function(A, ...) { e <- eigen(A, symmetric = TRUE); v <- e$vectors[, 1]; if (sum(v) < 0) v <- -v
  list(vector = v, value = e$values[1]) }
cl <- rep(1:3, each = 20)
Xc <- cbind(matrix(rnorm(60 * 3, rep(c(0, 3, 0), each = 20), 1), 60, 3),
            matrix(rnorm(60 * 3, rep(c(0, 0, 3), each = 20), 1), 60, 3))
Xc[, 2] <- Xc[, 1] + rnorm(60, 0, 0.5); Xc[, 5] <- Xc[, 4] + rnorm(60, 0, 0.5)
U0 <- matrix(0, 60, 3); U0[cbind(1:60, sample(rep(1:3, 20)))] <- 1
V0 <- matrix(0, 6, 2); V0[cbind(1:6, c(1, 2, 1, 2, 1, 2))] <- 1
invisible(capture.output(cd <- CDpca0(Xc, U0, V0, P = 3, Q = 2, tol = 1e-5, maxit = 100, r = 1, cdpcaplot = FALSE)))
out$cdpca <- list(X = Xc, U0 = U0, V0 = V0, F = cd$F, U = um(cd$U), V = um(cd$V), A = um(cd$A), iter = cd$iter,
                  bcdev = cd$bcdev, bcdevTotal = cd$bcdevTotal, Enorm = cd$Enorm, truth = cl)

## ---- mixmeta ----
data(berkey98)
mm <- list()
for (meth in c("reml", "ml", "fixed")) for (mods in c(FALSE, TRUE)) {
  f <- if (mods) cbind(PD, AL) ~ pubyear else cbind(PD, AL) ~ 1
  m <- mixmeta(f, S = berkey98[5:7], data = berkey98, method = meth, control = list(reltol = 1e-14, maxiter = 5000))
  mm[[paste(meth, mods, sep = "_")]] <- list(coef = unname(coef(m)), vcov = um(vcov(m)), Psi = if (meth == "fixed") NULL else um(m$Psi),
                                            logLik = as.numeric(logLik(m)), Q = unname(qtest(m)$Q[1]), dfQ = unname(qtest(m)$df[1]))
}
# larger example with 3 outcomes and missing values
ns <- 40; K <- 3
Psi_t <- matrix(c(0.3, 0.1, 0, 0.1, 0.2, 0.05, 0, 0.05, 0.4), 3); xm <- rnorm(ns)
Sl <- lapply(1:ns, function(i) { s <- runif(3, 0.05, 0.5); R <- matrix(0.3, 3, 3); diag(R) <- 1; diag(s) %*% R %*% diag(s) })
ys <- t(sapply(1:ns, function(i) c(1, -0.5, 0.2) + 0.4 * xm[i] + mvrnorm(1, rep(0, 3), Psi_t) + mvrnorm(1, rep(0, 3), Sl[[i]])))
ys[sample(ns, 6), 2] <- NA; ys[sample(ns, 5), 3] <- NA
Smat <- t(sapply(Sl, function(s) s[lower.tri(s, diag = TRUE)]))
m3 <- mixmeta(ys ~ xm, S = Smat, method = "reml", control = list(reltol = 1e-14, maxiter = 5000))
mm$sim3 <- list(y = ys, S = lapply(Sl, um), x = xm, coef = unname(coef(m3)), vcov = um(vcov(m3)), Psi = um(m3$Psi),
                logLik = as.numeric(logLik(m3)))
out$mixmeta <- c(list(y = um(berkey98[, c("PD", "AL")]), S = lapply(1:5, function(i) { v <- unlist(berkey98[i, 5:7]); matrix(c(v[1], v[2], v[2], v[3]), 2) }),
                      pubyear = berkey98$pubyear), mm)

write_json(out, "/home/claude/py/biplotpy/tests/fixtures/phase3.json", digits = NA, auto_unbox = TRUE, matrix = "rowmajor", null = "null")
cat("ok\n")
