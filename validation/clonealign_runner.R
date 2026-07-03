#!/usr/bin/env Rscript
# Run the REAL clonealign (kieranrcampbell/clonealign) on iscc-simulated data.
#
# This runner lives on the far side of the `iscc-clonealign` conda env (R + TensorFlow + the genuine
# clonealign package): the iscc validation script generates the tumour and writes the inputs, this
# script runs clonealign, and the results are read back for scoring. Keeping clonealign in its own env
# means the core `iscc` env never carries the heavy R/TF stack.
#
# We use clonealign's own `run_clonealign` multi-restart wrapper (several initial shrinks x repeats,
# keeping the best-ELBO fit): the variational objective is sensitive to initialisation, so a single
# run can land in a poor local optimum — the restart wrapper is the package's intended entry point.
#
# Inputs  (indir):  Y.csv  cells x genes scRNA counts (rownames=cells, colnames=genes)
#                   L.csv  genes x clones copy number  (rownames=genes, colnames=clones)
# Outputs (outdir): clone_probs.csv  cells x clones assignment probabilities
#                   clone_call.csv    cells -> hard clone call (argmax over clones)
#
# Usage: Rscript clonealign_runner.R <indir> <outdir> [max_iter] [n_repeats] [seed]

# Point reticulate at THIS conda env's python (where tensorflow + tensorflow_probability live),
# not reticulate's auto-provisioned uv python.
local({
  env_py <- file.path(dirname(dirname(R.home())), "bin", "python")
  if (file.exists(env_py)) Sys.setenv(RETICULATE_PYTHON = env_py)
})
suppressWarnings(suppressMessages({
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")
  library(clonealign)
  library(tensorflow)
}))

args <- commandArgs(trailingOnly = TRUE)
indir  <- args[1]
outdir <- args[2]
max_iter  <- if (length(args) >= 3) as.integer(args[3]) else 200L
n_repeats <- if (length(args) >= 4) as.integer(args[4]) else 3L
seed      <- if (length(args) >= 5) as.integer(args[5]) else 1L
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ---- load inputs ----
Y <- as.matrix(read.csv(file.path(indir, "Y.csv"), row.names = 1, check.names = FALSE))  # cells x genes
L <- as.matrix(read.csv(file.path(indir, "L.csv"), row.names = 1, check.names = FALSE))  # genes x clones

# clonealign matches genes between colnames(Y) and rownames(L).
common <- intersect(colnames(Y), rownames(L))
Y <- Y[, common, drop = FALSE]; storage.mode(Y) <- "double"
L <- L[common, , drop = FALSE]; storage.mode(L) <- "double"

set.seed(seed)
tf$compat$v1$set_random_seed(seed)
cat(sprintf("clonealign: %d cells x %d genes, %d clones (max_iter=%d, n_repeats=%d)\n",
            nrow(Y), ncol(Y), ncol(L), max_iter, n_repeats))

fit <- run_clonealign(Y, L, initial_shrinks = c(0, 5, 10), n_repeats = n_repeats,
                      print_elbos = FALSE, max_iter = max_iter, verbose = FALSE)

probs <- as.data.frame(fit$ml_params$clone_probs)
colnames(probs) <- colnames(L)
rownames(probs) <- rownames(Y)
write.csv(probs, file.path(outdir, "clone_probs.csv"))
write.csv(data.frame(clone = fit$clone, row.names = rownames(Y)),
          file.path(outdir, "clone_call.csv"))
cat("clonealign done ->", outdir, "\n")
