#!/usr/bin/env Rscript
# Splatter or scDesign3 fitted to Fig 3's scRNA reference (a breast tumour of Wu et al. 2021), for the
# data-simulator supplement (supp_datasim.py).
#
# Reads <in>/counts.mtx (genes x cells, written by `supp_datasim.py prep-scrna`) and <in>/cluster.txt,
# fits the simulator in its own published way and simulates as many cells and genes as the input:
#   splatter   splatEstimate() on the counts, then splatSimulate() with the estimate (defaults otherwise:
#              one group, dropout.type "none").
#   scdesign3  scdesign3() as its cell-type tutorial: negative binomial marginals with mean by cluster,
#              one dispersion, Gaussian copula by cluster.
# Writes <out>/<sim>.mtx (genes x cells) and <out>/<sim>.json (fitted parameters, runtime).
#
# Usage: Rscript datasim_scrna.R <in_dir> <out_dir> <splatter|scdesign3> [n_cores] [seed]
suppressMessages({library(Matrix); library(SingleCellExperiment); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
ind <- args[1]; out <- args[2]; sim <- args[3]
cores <- if (length(args) >= 4) as.integer(args[4]) else 1L
seed <- if (length(args) >= 5) as.integer(args[5]) else 1L
counts <- as(readMM(file.path(ind, "counts.mtx")), "CsparseMatrix")
rownames(counts) <- paste0("g", seq_len(nrow(counts)))
colnames(counts) <- paste0("c", seq_len(ncol(counts)))
cl <- factor(paste0("k", readLines(file.path(ind, "cluster.txt"))))
cat("input", nrow(counts), "genes x", ncol(counts), "cells;", nlevels(cl), "clusters\n")
set.seed(seed)
t0 <- Sys.time()
if (sim == "splatter") {
  suppressMessages(library(splatter))
  params <- splatEstimate(as.matrix(counts))
  print(params)
  res <- splatSimulate(params, seed = seed, verbose = FALSE)
  new <- counts(res)
  info <- list(package = paste("splatter", packageVersion("splatter")),
               fitted = lapply(setNames(nm = c("mean.shape", "mean.rate", "lib.loc", "lib.scale", "out.prob",
                                              "out.facLoc", "out.facScale", "bcv.common", "bcv.df",
                                              "dropout.type", "dropout.mid", "dropout.shape")),
                               function(p) getParam(params, p)))
} else if (sim == "scdesign3") {
  suppressMessages(library(scDesign3))
  sce <- SingleCellExperiment(list(counts = counts))
  colData(sce)$cell_type <- cl
  res <- scdesign3(sce = sce, assay_use = "counts", celltype = "cell_type", pseudotime = NULL,
                   spatial = NULL, other_covariates = NULL, mu_formula = "cell_type", sigma_formula = "1",
                   family_use = "nb", n_cores = cores, usebam = FALSE, corr_formula = "cell_type",
                   copula = "gaussian", DT = TRUE, pseudo_obs = FALSE, return_model = FALSE,
                   nonzerovar = FALSE)
  new <- res$new_count
  info <- list(package = paste("scDesign3", packageVersion("scDesign3")),
               settings = "mu ~ cell_type, sigma ~ 1, nb, copula gaussian by cell_type, DT, n_cores")
} else stop("unknown simulator ", sim)
info$seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
info$n_genes <- nrow(new); info$n_cells <- ncol(new); info$seed <- seed
cat(sim, "done in", info$seconds, "s:", nrow(new), "x", ncol(new), "\n")
writeMM(as(new, "CsparseMatrix"), file.path(out, paste0(sim, ".mtx")))
writeLines(toJSON(info, auto_unbox = TRUE, digits = NA), file.path(out, paste0(sim, ".json")))
