# cardelino (McCarthy et al. 2020) on one set of scRNA allele counts, for the SNV supplement of paper 1.
#
#   Rscript cardelino_runner.R IN_DIR OUT_DIR MIN_ITER MAX_ITER SEED
#
# IN_DIR holds config.csv (variants x clones, 0/1, variant names in the first column), cells.txt (one
# cell id per line) and ad.tsv (columns variant, cell, A, D: the alt and total reads of each covered
# variant-cell pair, cell a 0-based index into cells.txt; pairs not listed have no reads). Runs
# clone_id with the given Config and cardelino's defaults otherwise (Gibbs sampling, relax_Config on,
# uniform clone prior), the iteration counts of the package vignette unless given. Writes
# OUT_DIR/prob.csv (cells x clones, the posterior probability of each clone) and OUT_DIR/run.json.
suppressMessages({
  library(cardelino)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) stop("usage: cardelino_runner.R IN_DIR OUT_DIR MIN_ITER MAX_ITER SEED")
in_dir <- args[1]; out_dir <- args[2]
min_iter <- as.integer(args[3]); max_iter <- as.integer(args[4]); seed <- as.integer(args[5])
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

config <- as.matrix(read.csv(file.path(in_dir, "config.csv"), row.names = 1, check.names = FALSE))
storage.mode(config) <- "double"
cells <- readLines(file.path(in_dir, "cells.txt"))
ad <- read.delim(file.path(in_dir, "ad.tsv"), stringsAsFactors = FALSE,
                 colClasses = c("character", "integer", "integer", "integer"))
A <- matrix(0, nrow(config), length(cells), dimnames = list(rownames(config), cells))
D <- A
if (nrow(ad) > 0) {
  vi <- match(ad$variant, rownames(config))
  if (anyNA(vi)) stop("ad.tsv names a variant missing from config.csv")
  ci <- ad$cell + 1L
  A[cbind(vi, ci)] <- ad$A
  D[cbind(vi, ci)] <- ad$D
}
cat(sprintf("cardelino: %d variants x %d cells x %d clones; %d covered pairs; %d cells with no read\n",
            nrow(A), ncol(A), ncol(config), sum(D > 0), sum(colSums(D) == 0)))

set.seed(seed)
t0 <- proc.time()[["elapsed"]]
res <- clone_id(A, D, Config = config, inference = "sampling", min_iter = min_iter, max_iter = max_iter,
                verbose = FALSE)
wall <- proc.time()[["elapsed"]] - t0

prob <- res$prob
rownames(prob) <- cells
colnames(prob) <- colnames(config)
write.csv(prob, file.path(out_dir, "prob.csv"), quote = TRUE)

num <- function(x) if (is.null(x)) NULL else as.numeric(x)
# cardelino's own convergence check (clone_id_Gibbs): the share of the cell x clone probability traces whose
# Geweke Z is within 2; the sampler stops early only when it exceeds 99.5%.
conv <- tryCatch(abs(cardelino:::Geweke_Z(res$prob_all)) <= 2, error = function(e) NA)
info <- list(
  cardelino = as.character(packageVersion("cardelino")),
  n_variants = nrow(A), n_cells = ncol(A), n_clones = ncol(config),
  n_covered_pairs = sum(D > 0), n_cells_no_read = sum(colSums(D) == 0),
  min_iter = min_iter, max_iter = max_iter, seed = seed, wall_s = round(wall, 1),
  result_fields = names(res),
  relax_rate = num(res$relax_rate),
  theta0_mean = if (is.null(res$theta0)) NULL else mean(as.numeric(res$theta0)),
  theta1_mean = if (is.null(res$theta1)) NULL else mean(as.numeric(res$theta1)),
  n_iter = if (is.null(res$prob_all)) NULL else nrow(res$prob_all),
  percent_converged = if (all(is.na(conv))) NULL else round(100 * mean(conv, na.rm = TRUE), 2)
)
writeLines(toJSON(info, auto_unbox = TRUE, null = "null", digits = 6), file.path(out_dir, "run.json"))
cat(sprintf("cardelino: done in %.1f s\n", wall))
