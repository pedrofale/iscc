# SpatialSimBench's spot- and gene-level metrics (Liang et al. 2025, Genome Biology) for one simulated
# section against one real section, each side computed on its own spots.
#
# Adapted from the OpenProblems port of SpatialSimBench, openproblems-bio/task_spatial_simulators v0.2.0,
# src/metrics/ks_statistic_gene_cell/script.R (MIT licence), which computes every statistic of the real
# and the simulated data separately (the original scripts assumed the simulation sits at the real spots).
# Changes: the inputs are two Matrix Market files (spots x genes) instead of AnnData, and each metric
# reports both ks::kde.test's z statistic (what the published SpatialSimBench results table holds) and its
# T statistic (what OpenProblems reports). The metric names follow the published table.
#
#   Rscript spatialsimbench_ks.R real.mtx simulated.mtx out.csv
suppressPackageStartupMessages({library(Matrix); library(matrixStats)})
args <- commandArgs(trailingOnly = TRUE)
real_counts <- as(readMM(args[1]), "CsparseMatrix")
sim_counts <- as(readMM(args[2]), "CsparseMatrix")
stopifnot(ncol(real_counts) == ncol(sim_counts))

as_finite_kde_input <- function(x) {
  if (is.data.frame(x)) x <- as.matrix(x)
  if (is.matrix(x)) {
    x <- matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x))
    return(x[apply(is.finite(x), 1, all), , drop = FALSE])
  }
  x <- as.numeric(x)
  x[is.finite(x)]
}
has_no_variance <- function(x) {
  if (is.matrix(x)) any(apply(x, 2, stats::sd, na.rm = TRUE) == 0) else stats::sd(x, na.rm = TRUE) == 0
}
add_kde_jitter <- function(x, amount) {
  x_range <- max(x, na.rm = TRUE) - min(x, na.rm = TRUE)
  x_scale <- max(x_range, stats::sd(as.numeric(x), na.rm = TRUE), 1)
  noise <- seq(-amount, amount, length.out = length(x)) * x_scale
  if (is.matrix(x)) matrix(as.numeric(x) + noise, nrow = nrow(x), ncol = ncol(x)) else as.numeric(x) + noise
}
missing_result <- function(reason) {
  warning(reason, " Returning NA for this metric.")
  list(zstat = NA_real_, Tstat = NA_real_)
}
try_kde_test <- function(x1, x2) {
  x1 <- as_finite_kde_input(x1)
  x2 <- as_finite_kde_input(x2)
  if (length(x1) == 0 || length(x2) == 0) return(missing_result("No finite values."))
  if (has_no_variance(x1) || has_no_variance(x2)) return(missing_result("No variance."))
  last_error <- NULL
  for (jitter in c(0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)) {
    a <- if (jitter == 0) x1 else add_kde_jitter(x1, jitter)
    b <- if (jitter == 0) x2 else add_kde_jitter(x2, jitter)
    res <- tryCatch(ks::kde.test(x1 = a, x2 = b), error = function(e) { last_error <<- e; NULL })
    if (!is.null(res)) return(res)
  }
  missing_result(paste0("ks::kde.test failed: ", last_error$message))
}
subsample <- function(x, max_size = 10000L, seed = 1L) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (length(x) <= max_size) return(x)
  set.seed(seed)
  sample(x, size = max_size, replace = FALSE)
}

out <- list()
add <- function(level, name, res) out[[length(out) + 1]] <<- data.frame(level = level, metric = name,
                                                                         zstat = res$zstat, Tstat = res$Tstat)

frac_zero_real_genes <- colMeans(real_counts == 0)
frac_zero_sim_genes <- colMeans(sim_counts == 0)
frac_zero_real_spots <- rowMeans(real_counts == 0)
frac_zero_sim_spots <- rowMeans(sim_counts == 0)
lib_real <- log1p(rowSums(real_counts))
lib_sim <- log1p(rowSums(sim_counts))
real_dge <- edgeR::calcNormFactors(edgeR::DGEList(counts = Matrix::t(real_counts)), method = "TMM")
sim_dge <- edgeR::calcNormFactors(edgeR::DGEList(counts = Matrix::t(sim_counts)), method = "TMM")

# gene level
add("gene", "scaledVar", try_kde_test(as.numeric(scale(sparseMatrixStats::colVars(real_counts))),
                                      as.numeric(scale(sparseMatrixStats::colVars(sim_counts)))))
add("gene", "scaledMean", try_kde_test(as.numeric(scale(colMeans(real_counts))), as.numeric(scale(colMeans(sim_counts)))))
add("gene", "pearson", try_kde_test(subsample(proxyC::simil(real_counts, margin = 2, method = "correlation")),
                                    subsample(proxyC::simil(sim_counts, margin = 2, method = "correlation"))))
add("gene", "MeanVar", try_kde_test(data.frame(mean = colMeans(real_counts), var = sparseMatrixStats::colVars(real_counts)),
                                    data.frame(mean = colMeans(sim_counts), var = sparseMatrixStats::colVars(sim_counts))))
add("gene", "MeanFrac", try_kde_test(data.frame(mean = colMeans(real_counts), fraczero = frac_zero_real_genes),
                                     data.frame(mean = colMeans(sim_counts), fraczero = frac_zero_sim_genes)))
add("gene", "fracZero", try_kde_test(frac_zero_real_genes, frac_zero_sim_genes))
# spot level
add("spot", "TMM", try_kde_test(real_dge$samples$norm.factors, sim_dge$samples$norm.factors))
add("spot", "scaledVar", try_kde_test(as.numeric(scale(sparseMatrixStats::colVars(Matrix::t(real_counts)))),
                                      as.numeric(scale(sparseMatrixStats::colVars(Matrix::t(sim_counts))))))
add("spot", "scaledMean", try_kde_test(as.numeric(scale(rowMeans(real_counts))), as.numeric(scale(rowMeans(sim_counts)))))
add("spot", "pearson", try_kde_test(subsample(proxyC::simil(real_counts, margin = 1, method = "correlation")),
                                    subsample(proxyC::simil(sim_counts, margin = 1, method = "correlation"))))
add("spot", "libsize", try_kde_test(lib_real, lib_sim))
add("spot", "libFracZero", try_kde_test(data.frame(lib = lib_real, fraczero = frac_zero_real_spots),
                                        data.frame(lib = lib_sim, fraczero = frac_zero_sim_spots)))
add("spot", "fracZero", try_kde_test(frac_zero_real_spots, frac_zero_sim_spots))
add("spot", "EffLibsize", try_kde_test(log1p(real_dge$samples$lib.size * real_dge$samples$norm.factors),
                                       log1p(sim_dge$samples$lib.size * sim_dge$samples$norm.factors)))
write.csv(do.call(rbind, out), args[3], row.names = FALSE)
