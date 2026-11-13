## Fixtures for phase 4 (biplots ported from R): CA, MCA, CCA, RDA, PCoA, MDS, external logistic,
## PLS, FA, NIPALS, co-inertia, canonical distance analysis, polychoric biplot.
suppressPackageStartupMessages({
  library(MultBiplotR); library(ca); library(FactoMineR); library(vegan); library(ade4); library(smacof)
  library(pls); library(psych); library(jsonlite)
})
out <- list(); um <- function(m) { m <- as.matrix(m); attributes(m) <- list(dim = dim(m)); m }

## ---- CA ----
data(smoke, package = "ca")
out$ca <- list(N = um(smoke))
for (a in c(1, 0, 0.5)) { m <- MultBiplotR::CA(smoke, dim = 2, alpha = a)
  out$ca[[paste0("mbr_", a)]] <- list(rows = um(m$RowCoordinates), cols = um(m$ColCoordinates), eig = m$EigenValues,
                                     rq = um(m$RowContributions), cq = um(m$ColContributions)) }
cc <- ca::ca(smoke)
out$ca$ca_pkg <- list(sv = cc$sv, rowcoord = um(cc$rowcoord), colcoord = um(cc$colcoord), rowmass = cc$rowmass,
                      colmass = cc$colmass, ctr_r = um(cbind(cc$rowmass * (cc$rowcoord[, 1] * cc$sv[1])^2 / cc$sv[1]^2,
                                                             cc$rowmass * (cc$rowcoord[, 2] * cc$sv[2])^2 / cc$sv[2]^2)))

## ---- MCA ----
data(poison, package = "FactoMineR")
po <- poison[, 5:15]
mc <- MCA(po, ncp = 3, graph = FALSE)
out$mca <- list(X = lapply(po, as.character), eig = unname(mc$eig[, 1]), ind = um(mc$ind$coord), var = um(mc$var$coord),
                var_names = rownames(mc$var$coord), eta2 = um(mc$var$eta2))

## ---- CCA and RDA (vegan) ----
data(varespec); data(varechem)
env <- varechem[, c("N", "P", "K", "Ca", "Al")]
mc2 <- MultBiplotR::CCA(varespec, env, alpha = 1, dimens = 3)
vc <- vegan::cca(varespec, env)
out$cca <- list(Y = um(varespec), Z = um(env), eig = mc2$EigenValues, species = um(mc2$Species_Scores),
                sites = um(mc2$Site_Scores), lc = um(mc2$LC_Site_Scores), env = um(mc2$Env_Var_Scores),
                vegan_eig = unname(vc$CCA$eig), vegan_tot = vc$tot.chi)
vr <- vegan::rda(varespec, env)
out$rda <- list(eig = unname(vr$CCA$eig), ca_eig = unname(vr$CA$eig), tot = vr$tot.chi)
for (s in 1:3) {
  sc <- vegan::scores(vr, choices = 1:2, display = c("sp", "wa", "lc", "bp"), scaling = s)
  out$rda[[paste0("s", s)]] <- list(species = um(sc$species), sites = um(sc$sites), lc = um(sc$constraints), bp = um(sc$biplot))
}
vr2 <- vegan::rda(varespec, env, scale = TRUE)
sc <- vegan::scores(vr2, choices = 1:2, display = c("sp", "wa", "bp"), scaling = 2)
out$rda$scaled <- list(eig = unname(vr2$CCA$eig), species = um(sc$species), sites = um(sc$sites), bp = um(sc$biplot))

## ---- PCoA (MultBiplotR, ade4::dudi.pco) ----
X <- as.matrix(iris[1:40, 1:4])
pr <- ContinuousProximities(X, coef = 1)
pco <- PrincipalCoordinates(pr, dimension = 3)
dp <- dudi.pco(dist(X), scannf = FALSE, nf = 3)
set.seed(1); Bin <- matrix(rbinom(30 * 8, 1, 0.45), 30, 8); colnames(Bin) <- paste0("b", 1:8); rownames(Bin) <- paste0("r", 1:30)
bprox <- BinaryProximities(Bin, coefficient = "Jaccard")
bpco <- PrincipalCoordinates(bprox, dimension = 2)
ext <- ExternalBinaryLogisticBiplot(bpco, penalization = 0.2)
out$pcoa <- list(X = X, eig = pco$EigenValues, coords = um(pco$RowCoordinates), ade4_li = um(dp$li), ade4_eig = dp$eig,
                 stress1 = pco$stress1, rsq = pco$rsq,
                 Bin = Bin, jaccard = um(bprox$Proximities), bin_coords = um(bpco$RowCoordinates),
                 ext_params = um(ext$ColumnParameters), ext_dev = unname(ext$VarInfo$Deviances),
                 ext_pct = unname(ext$VarInfo$PercentsCorrec))
bc <- list()
for (cf in c("Russell_and_Rao", "Simple_Matching", "Sorensen_Dice_and_Czekanowski", "Ochiai", "Rogers_and_Tanimoto", "Sneath_and_Sokal", "Anderberg"))
  bc[[cf]] <- um(BinaryDistances(Bin, coefficient = cf, transformation = "sqrt(1-S)"))
out$pcoa$binary <- bc

## ---- MDS (smacof::mds) ----
Dm <- round(as.matrix(dist(X)), 8)   # rounded so that ties survive the JSON round trip
md <- list()
for (ty in c("ratio", "interval", "ordinal")) {
  m <- smacof::mds(Dm, ndim = 2, type = ty)
  md[[ty]] <- list(conf = um(m$conf), stress = m$stress, niter = m$niter, dhat = um(as.matrix(m$dhat)))
}
m <- smacof::mds(Dm, ndim = 2, type = "ordinal", ties = "secondary"); md$ordinal_s <- list(conf = um(m$conf), stress = m$stress, niter = m$niter)
out$mds <- c(list(D = um(Dm)), md)

## ---- PLS (pls::plsr) ----
data(yarn, package = "pls")
data(oliveoil, package = "pls")
oy <- as.matrix(oliveoil$sensory); ox <- as.matrix(oliveoil$chemical)
pm <- plsr(oy ~ ox, ncomp = 3, scale = TRUE, method = "oscorespls")
out$pls <- list(X = um(ox), Y = um(oy), scores = um(pls::scores(pm)), loadings = um(pls::loadings(pm)), yloadings = um(pls::Yloadings(pm)),
                weights = um(pls::loading.weights(pm)), coef = um(coef(pm, ncomp = 3)[, , 1]), fitted = um(fitted(pm)[, , 3]))

## ---- FA biplot (MultBiplotR, psych) ----
fa1 <- FA.Biplot(as.matrix(iris[, 1:4]), dimension = 2, Extraction = "PC", Rotation = "varimax")
fa2 <- FA.Biplot(as.matrix(iris[, 1:4]), dimension = 2, Extraction = "PC", Rotation = "none")
pp <- psych::principal(iris[, 1:4], nfactors = 2, rotate = "varimax")
out$fa <- list(rows_vm = um(fa1$RowCoordinates), cols_vm = um(fa1$ColCoordinates), comm = unname(fa1$Communalities),
               rows_none = um(fa2$RowCoordinates), cols_none = um(fa2$ColCoordinates), ev = unname(fa2$EigenValues),
               psych_loadings = um(unclass(pp$loadings)))

## ---- NIPALS with missing values (ade4::nipals) ----
set.seed(3); Xn <- as.matrix(scale(iris[1:60, 1:4])); Xmiss <- Xn; Xmiss[sample(length(Xn), 15)] <- NA
nm <- ade4::nipals(as.data.frame(Xmiss), nf = 2, rec = TRUE)
nf <- ade4::nipals(as.data.frame(Xn), nf = 2)
out$nipals <- list(Xmiss = Xmiss, Xfull = Xn, li = um(nm$li), c1 = um(nm$c1), eig = nm$eig, rec = um(nm$rec),
                   full_li = um(nf$li), full_eig = nf$eig, full_c1 = um(nf$c1))
mb <- NIPALS.Biplot(as.matrix(iris[, 1:4]), alpha = 1, dimension = 2)
out$nipals$mbr_rows <- um(mb$RowCoordinates); out$nipals$mbr_cols <- um(mb$ColCoordinates)

## ---- Co-inertia (ade4) ----
data(doubs, package = "ade4")
d1 <- dudi.pca(doubs$env, scannf = FALSE, nf = 3); d2 <- dudi.pca(doubs$fish, scannf = FALSE, nf = 3, scale = FALSE)
co <- coinertia(d1, d2, scannf = FALSE, nf = 2)
out$coin <- list(X = um(doubs$env), Y = um(doubs$fish), eig = co$eig, RV = co$RV, c1 = um(co$c1), l1 = um(co$l1),
                 co = um(co$co), li = um(co$li), lX = um(co$lX), lY = um(co$lY))

## ---- Canonical distance analysis (MultBiplotR) ----
pri <- ContinuousProximities(as.matrix(iris[, 1:4]), coef = 1)
set.seed(4); cda <- CanonicalDistanceAnalysis(pri, iris$Species, dimens = 2, Nsamples = 200)
out$cda <- list(F = cda$Fexp, TSS = cda$TSS, BSS = cda$BSS, means = um(cda$MeanCoordinates), rows = um(cda$RowCoordinates),
                expl = cda$ExplainedVariance)

## ---- Polychoric correlations (psych) ----
set.seed(5); lat <- MASS::mvrnorm(300, rep(0, 4), matrix(c(1, .6, .5, .3, .6, 1, .4, .2, .5, .4, 1, .3, .3, .2, .3, 1), 4))
Ord <- sapply(1:4, function(j) cut(lat[, j], c(-Inf, -0.8, 0, 0.9, Inf), labels = FALSE))
pc <- psych::polychoric(Ord)
out$poly <- list(X = Ord, rho = um(pc$rho), tau = um(pc$tau))

write_json(out, "/home/claude/py/biplotpy/tests/fixtures/phase4.json", digits = NA, auto_unbox = TRUE, matrix = "rowmajor")
cat("ok\n")
## ---- extra: factor scores (psych) and GPArotation references ----
out <- jsonlite::fromJSON("/home/claude/py/biplotpy/tests/fixtures/phase4.json", simplifyVector = FALSE)
library(GPArotation)
L0 <- unclass(psych::principal(iris[, 1:4], nfactors = 2, rotate = "none")$loadings)
rots <- list()
for (m in c("varimax", "quartimax", "entropy")) rots[[m]] <- um(GPForth(L0, method = m, normalize = FALSE)$loadings)
rots$equamax <- um(GPForth(L0, method = "cf", methodArgs = list(kappa = 2 / (2 * 4)))$loadings)
rots$quartimin <- um(GPFoblq(L0, method = "quartimin")$loadings)
rots$oblimin <- um(GPFoblq(L0, method = "oblimin", methodArgs = list(gam = 0.5))$loadings)
set.seed(9); L5 <- matrix(runif(24, -1, 1), 8, 3)
rots5 <- list(L = L5, varimax = um(GPForth(L5, method = "varimax")$loadings), quartimin = um(GPFoblq(L5, method = "quartimin")$loadings),
              varimax_norm = um(GPForth(L5, method = "varimax", normalize = TRUE)$loadings))
Zs <- scale(as.matrix(iris[, 1:4]))
fsB <- psych::factor.scores(Zs, f = um(fa1$ColCoordinates), method = "Bartlett")$scores
fsT <- psych::factor.scores(Zs, f = um(fa1$ColCoordinates), method = "Thurstone")$scores
out$fa$L0 <- um(L0); out$fa$rot <- rots; out$fa$rot5 <- rots5; out$fa$bartlett <- um(fsB); out$fa$thurstone <- um(fsT)
write_json(out, "/home/claude/py/biplotpy/tests/fixtures/phase4.json", digits = NA, auto_unbox = TRUE, matrix = "rowmajor")
cat("ok2\n")
