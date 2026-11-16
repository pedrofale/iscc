#!/usr/bin/env Rscript
# Splatter's version of the Ding et al. A-B contrast, for batch_validation.py (Fig 5's batch panel,
# registered in exp-paper1-fig5-cohort, "Amendment, 2026-11-16 night"). Runs in the datasim-r image
# (Splatter 1.34.0; Zappia, Phipson & Oshlack 2017, Genome Biol 18:174, doi:10.1186/s13059-017-1305-0),
# never in envs/iscc.
#
#   Rscript batch_splatter.R estimate <counts.bin> <params.rds> <info.json>
#       splatEstimate() on the pooled real counts (genes x cells, the binary format below).
#   Rscript batch_splatter.R simulate <params.rds> <spec.json> <out_dir>
#       one splatSimulate(method = "groups") per run of the spec, each written to
#       <out_dir>/<name>.bin and <out_dir>/<name>_cols.tsv (Batch, Group per cell).
#
# The spec (JSON) is {"group_prob": [...], "batch_cells": [nA, nB], "runs": [{"name", "seed",
# "de_facLoc", "batch_facScale"}, ...]}. Every run keeps splatEstimate's parameters and Splatter's
# defaults otherwise (de.prob 0.1, de.downProb 0.5, de.facScale 0.4, dropout.type "none");
# batch.facLoc is 0 as registered, batch.facScale and de.facLoc come from the run.
#
# Binary format (little-endian int32): nrow, ncol, nnz, then p (ncol + 1), i (nnz, 0-based), x (nnz):
# a column-compressed genes x cells count matrix.
suppressMessages({library(Matrix); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]

read_bin <- function(path) {
  con <- file(path, "rb"); on.exit(close(con))
  d <- readBin(con, "integer", 3, size = 4, endian = "little")
  p <- readBin(con, "integer", d[2] + 1, size = 4, endian = "little")
  i <- readBin(con, "integer", d[3], size = 4, endian = "little")
  x <- readBin(con, "integer", d[3], size = 4, endian = "little")
  sparseMatrix(i = i, p = p, x = as.numeric(x), dims = d[1:2], index1 = FALSE)
}

write_bin <- function(m, path) {
  m <- as(m, "CsparseMatrix")
  m <- as(m, "dgCMatrix")
  con <- file(path, "wb"); on.exit(close(con))
  writeBin(as.integer(c(nrow(m), ncol(m), length(m@x))), con, size = 4, endian = "little")
  writeBin(as.integer(m@p), con, size = 4, endian = "little")
  writeBin(as.integer(m@i), con, size = 4, endian = "little")
  writeBin(as.integer(round(m@x)), con, size = 4, endian = "little")
}

suppressMessages(library(splatter))
if (mode == "estimate") {
  counts <- read_bin(args[2])
  cat("estimate on", nrow(counts), "genes x", ncol(counts), "cells\n")
  t0 <- Sys.time()
  params <- splatEstimate(as.matrix(counts))
  saveRDS(params, args[3])
  keys <- c("mean.shape", "mean.rate", "lib.loc", "lib.scale", "lib.norm", "out.prob", "out.facLoc",
            "out.facScale", "bcv.common", "bcv.df", "dropout.type", "dropout.mid", "dropout.shape",
            "de.prob", "de.downProb", "de.facLoc", "de.facScale", "nGenes")
  info <- list(package = paste("splatter", packageVersion("splatter")), R = R.version.string,
               seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")),
               estimated = lapply(setNames(nm = keys), function(k) getParam(params, k)))
  writeLines(toJSON(info, auto_unbox = TRUE, digits = NA), args[4])
  print(params)
} else if (mode == "simulate") {
  params <- readRDS(args[2])
  spec <- fromJSON(args[3])
  out <- args[4]
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  runs <- spec$runs
  for (k in seq_len(nrow(runs))) {
    r <- runs[k, ]
    t0 <- Sys.time()
    p <- setParams(params, batchCells = as.integer(spec$batch_cells), batch.facLoc = 0,
                   batch.facScale = as.numeric(r$batch_facScale), group.prob = as.numeric(spec$group_prob),
                   de.facLoc = as.numeric(r$de_facLoc), seed = as.integer(r$seed))
    sim <- splatSimulate(p, method = "groups", verbose = FALSE)
    write_bin(counts(sim), file.path(out, paste0(r$name, ".bin")))
    cd <- as.data.frame(colData(sim))[, c("Batch", "Group")]
    write.table(cd, file.path(out, paste0(r$name, "_cols.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
    cat(r$name, ": seed", r$seed, "de.facLoc", r$de_facLoc, "batch.facScale", r$batch_facScale, "->",
        nrow(sim), "x", ncol(sim), "in", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "s\n")
    rm(sim); gc(verbose = FALSE)
  }
} else stop("unknown mode ", mode)
