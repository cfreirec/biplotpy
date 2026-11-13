suppressMessages({library(jsonlite); library(canbiplotME); library(MultBiplotR)})
out <- "fixtures"
wj <- function(x, f) write_json(x, file.path(out, f), digits = NA, auto_unbox = TRUE, matrix = "rowmajor", na = "null")
arr3 <- function(a) lapply(seq_len(dim(a)[1]), function(i) a[i, , ])
# ---------------- canbiplotME ----------------
data(sim_me)
wj(list(beta = unname(sim_me$beta), V = lapply(seq_len(sim_me$N), function(i) unname(sim_me$V[i, , ])),
        group = as.character(sim_me$group), id = sim_me$id, comps = colnames(sim_me$beta),
        D = lapply(sim_me$D, unname)), "sim_me.json")
fitout <- function(f) list(T = unname(f$T), T_raw = unname(f$T_raw), mu = unname(f$mu),
  mu_cov = arr3(unname(f$mu_cov)), mu0 = unname(f$mu0), W = unname(f$W), H = unname(f$H),
  lambda = f$lambda, V = unname(f$V), A = unname(f$A), Afull = unname(f$Afull), C = unname(f$C),
  cos2 = unname(f$cos2), quality = unname(f$quality), ind = unname(f$ind), ind_ids = rownames(f$ind),
  stats = as.list(f$stats), D2 = unname(f$D2), mbar = unname(f$mbar), n = f$n)
f1 <- cbme(sim_me, S_vars = c("z1", "z2"))
xi <- me_data(sim_me$beta, sim_me$V, sim_me$group)
f2 <- cbme(xi)
f3 <- cbme(sim_me, metric = "median", weights = "equal", T_errors = "joint")
f4 <- cbme(sim_me, metric = "group", mean_errors = "independent")
rg <- cbme_regions(f1, types = c("R1", "R2", "R3"))
reg <- lapply(rg$regions, function(r) lapply(r, function(z) list(S = unname(z$S), q = z$q, area = z$area)))
jr <- canbiplotME:::jack_rho(f1, sim_me$beta)
wj(list(joint = fitout(f1), independent = fitout(f2), median_equal_jointT = fitout(f3), group_metric = fitout(f4),
        regions = reg, jack_rho = jr), "cbme_fits.json")
# first stage
data(sim_trade)
write.csv(sim_trade, file.path(out, "sim_trade.csv"), row.names = FALSE)
s1 <- gravity_by_individual(sim_trade, "flow", "orig", "dest", "year", c("ldist", "lang", "fta"), "regime", controls = "contig")
cnt <- support_counts(sim_trade, "orig", "dest", c("lang", "fta"))
# one deterministic resample of pairs for group R1
d <- sim_trade[sim_trade$regime == "R1", ]; d$.pair <- paste(d$orig, d$dest, sep = "_")
pairs <- unique(d$.pair); set.seed(99); s <- sample(pairs, length(pairs), replace = TRUE)
rows <- unlist(split(seq_len(nrow(d)), d$.pair)[s], use.names = FALSE)
db <- d[rows, ]
m <- fixest::fepois(canbiplotME:::stage1_formula(s1$spec), data = db, vcov = "iid", glm.iter = 100, notes = FALSE)
rb <- canbiplotME:::extract_individual(coef(m), NULL, "orig", sort(unique(d$orig)), c("ldist", "lang", "fta"))$beta
dyn <- gravity_dynamic(sim_trade, "flow", "orig", "dest", "year", c("ldist", "lang", "fta"), "regime", controls = "contig")
wj(list(beta = unname(s1$beta), id = s1$id, group = s1$group, V = lapply(seq_len(nrow(s1$beta)), function(i) unname(s1$V[i, , ])),
        counts = cnt, resample_pairs = s, resample_beta = unname(rb),
        delta = lapply(dyn$groups, function(e) unname(e$delta))), "stage1.json")
# ---------------- MultBiplotR ----------------
X <- as.matrix(iris[, 1:4]); grp <- iris$Species
mt <- as.matrix(mtcars[, c("mpg", "disp", "hp", "drat", "wt", "qsec")])
pb <- function(b) list(a = unname(b$RowCoordinates), b = unname(b$ColCoordinates), ev = b$EigenValues,
  inertia = b$Inertia, scf = b$Scale_Factor, structure = unname(b$Structure), rc = unname(b$RowContributions),
  cc = unname(b$ColContributions), scaled = unname(b$Scaled_Data))
res <- list()
for (sc in c(4, 5)) for (al in c(0, 0.5, 1)) res[[sprintf("iris_s%d_a%s", sc, al)]] <- pb(PCA.Biplot(X, alpha = al, dimension = 2, Scaling = sc))
res[["iris_HJ_s5"]] <- pb(HJ.Biplot(X, dimension = 3, Scaling = 5))
res[["mtcars_HJ_s5"]] <- pb(HJ.Biplot(mt, dimension = 3, Scaling = 5))
res[["mtcars_s4_a1"]] <- pb(PCA.Biplot(mt, alpha = 1, dimension = 3, Scaling = 4))
res[["mtcars_s2_a1"]] <- pb(PCA.Biplot(mt, alpha = 1, dimension = 2, Scaling = 2))
res[["mtcars_s10_a1"]] <- pb(PCA.Biplot(mt, alpha = 1, dimension = 2, Scaling = 10))
res[["mtcars_s3_a1"]] <- pb(PCA.Biplot(mt, alpha = 1, dimension = 2, Scaling = 3))
# transforms
tr <- list(); for (t in c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)) tr[[as.character(t)]] <- unname(InitialTransform(mt, transform = t)$X)
# supplementary columns (scaled as in InitialTransform) and external continuous variables
b1 <- PCA.Biplot(X[, 1:3], alpha = 1, dimension = 2, Scaling = 5)
ac <- AddContVars2Biplot(b1, X[, 4, drop = FALSE], Scaling = 5)
sc5 <- GetBiplotScales(PCA.Biplot(mt, alpha = 1, dimension = 2, Scaling = 5), nticks = 4)
sc4 <- GetBiplotScales(PCA.Biplot(mt, alpha = 1, dimension = 2, Scaling = 4), nticks = 4)
# canonical biplot
cb <- CanonicalBiplot(X, group = grp, InitialTransform = 5)
cbo <- list(J = unname(cb$GroupCoordinates), H = unname(cb$ColCoordinates), B = unname(cb$Canonical_Weights),
  V = unname(cb$RowCoordinates), ev = cb$EigenValues, inertia = cb$Inertia, wilksf = cb$Wilksf, wilksp = cb$Wilksp,
  anovas = unname(cb$ANOVAS), univrad = unname(cb$UnivRad), bonfrad = unname(cb$BonfRad), multrad = unname(cb$MultRad),
  chisrad = unname(cb$ChisRad), groupcontr = unname(cb$GroupContributions), colcontr = unname(cb$ColContributions),
  qlrvars = unname(cb$QLRVars), rowcontr = unname(cb$RowContributions), expltotal = as.vector(cb$ExplTotal),
  structure = unname(cb$Structure_Correlations))
cb2 <- CanonicalBiplot(mt, group = factor(mtcars$cyl), InitialTransform = 4)
cbo2 <- list(J = unname(cb2$GroupCoordinates), H = unname(cb2$ColCoordinates), ev = cb2$EigenValues,
  wilksf = cb2$Wilksf, wilksp = cb2$Wilksp, anovas = unname(cb2$ANOVAS), bonfrad = unname(cb2$BonfRad))
wj(list(pca = res, transforms = tr, addcont = list(b0 = unname(ac$ContSupVarsBiplot$b0), coords = unname(ac$ContSupVarsBiplot$ColCoordinates),
  R2 = unname(ac$ContSupVarsBiplot$R2), structure = unname(ac$ContSupVarsBiplot$Structure)),
  scales5 = list(ticks = lapply(sc5$Ticks, unname), labels = lapply(sc5$Labels, unname)),
  scales4 = list(ticks = lapply(sc4$Ticks, unname), labels = lapply(sc4$Labels, unname)),
  canonical_iris = cbo, canonical_mtcars = cbo2), "multbiplotr.json")
write.csv(cbind(mtcars, name = rownames(mtcars)), file.path(out, "mtcars.csv"), row.names = FALSE)
write.csv(iris, file.path(out, "iris.csv"), row.names = FALSE)
cat("done\n")
