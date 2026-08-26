#!/usr/bin/env Rscript
# HMMcopy runner -- executed in the dedicated `iscc-hmmcopy` env, never from a notebook.
#
# Calls copy number per CELL from single-cell read depth, which is what a real scDNA pipeline does
# before any clone analysis. The clone CN profiles this yields are what clonealign consumes; taking
# them from iscc's own truth instead would make that benchmark vacuous.
#
# Writes the HMM states to <out_cell_cn.csv> and the corrected log2 depth to
# <out_cell_cn>_copy.csv, which is what the ploidy anchor is applied to.
#
# Usage:  Rscript hmmcopy_runner.R <reads.csv> <bins.csv> <out_cell_cn.csv>
suppressWarnings(suppressMessages(library(HMMcopy)))
suppressMessages(library(data.table))

args   <- commandArgs(trailingOnly = TRUE)
reads  <- read.csv(args[1], row.names = 1, check.names = FALSE)   # loci x cells
bins   <- read.csv(args[2], row.names = 1, check.names = FALSE)   # locus, chr, start, end, gc, map
stopifnot(all(rownames(reads) == rownames(bins)))

# HMMsegment's parameters must be derived from the CORRECTED data -- the template returned by
# HMMsegment(NULL, getparam=TRUE) has mu = NA and the fit dies with "missing value where TRUE/FALSE
# needed". So the per-cell loop asks for the parameters after correcting, which is the documented
# call anyway.
out  <- matrix(NA_real_, nrow = nrow(reads), ncol = ncol(reads),
               dimnames = list(rownames(reads), colnames(reads)))
# `copy` is the GC- and mappability-corrected log2 depth BEFORE discretisation. It is what a ploidy
# anchor is applied to: HMMcopy calls each cell's own modal state neutral, so on a whole-genome-
# doubled tumour every state comes out halved unless something outside the cell fixes the scale.
copyv <- out
ok <- 0
for (j in seq_len(ncol(reads))) {
  d <- data.table(chr = factor(bins$chr), start = as.integer(bins$start),
                  end = as.integer(bins$end), reads = as.integer(reads[[j]]),
                  gc = as.numeric(bins$gc), map = as.numeric(bins$map))
  res <- try({
    # mappability = 0.6, not the 0.9 default: iscc's modelled mappability spans ~0.33-1.0, and at
    # 0.9 too few bins survive the "ideal" filter for the GC loess (span = 0.03) to fit at all.
    corrected <- correctReadcount(d, mappability = 0.6, verbose = FALSE)
    seg <- HMMsegment(corrected, verbose = FALSE)
    list(state = seg$state, copy = corrected$copy)
  }, silent = TRUE)
  if (!inherits(res, "try-error") && length(res$state) == nrow(reads)) {
    out[, j]   <- as.numeric(res$state)
    copyv[, j] <- as.numeric(res$copy)
    ok <- ok + 1
  }
}
cat(sprintf("[hmmcopy_runner] called %d of %d cells over %d bins\n", ok, ncol(reads), nrow(reads)))
write.csv(out, args[3])
write.csv(copyv, sub("\\.csv$", "_copy.csv", args[3]))
