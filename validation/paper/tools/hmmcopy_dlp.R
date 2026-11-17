#!/usr/bin/env Rscript
# HMMcopy per cell with the DLP+ single-cell parameterisation (Laks et al. 2019; shahcompbio
# single_cell_pipeline, workflows/hmmcopy/scripts/hmmcopy_single_cell.R and config get_hmmcopy_params).
#
# HMMcopy's own defaults are made for bulk tumours: the six states start at quantiles of the cell's own
# log ratios, so they are ordinal (a one-copy gain can come back as "AMPL"), not copy numbers. The DLP+
# pipeline instead fixes twelve states at integer copy number 0..11 on the linear scale and, since a
# cell's ploidy is unknown, runs the HMM at multipliers 1..6 of the GC-corrected ratio, re-centres each
# run on its own states (the "tweak"), and keeps the multiplier whose copy values sit closest to integers
# (the lowest scaled "halfiness"; a run with more than 70% of bins at one copy is refused). This script
# follows that, with two differences stated here: the GC correction is HMMcopy's correctReadcount (loess)
# rather than the pipeline's modal regression, and the halfiness is measured on each bin's segment median.
#
# Usage: Rscript hmmcopy_dlp.R <reads.csv> <bins.csv> <out_prefix> [mappability]
#   reads.csv  loci x cells read counts; bins.csv  locus, chr, start, end, gc, map
# Writes <out_prefix>_state.csv (copy-number state per bin and cell, NA where not called),
#        <out_prefix>_cell.csv (cell, multiplier, scaled_halfiness, ok).
suppressWarnings(suppressMessages({library(HMMcopy); library(data.table)}))
args <- commandArgs(trailingOnly = TRUE)
reads <- read.csv(args[1], row.names = 1, check.names = FALSE)
bins <- read.csv(args[2], row.names = 1, check.names = FALSE)
prefix <- args[3]
map_cut <- if (length(args) >= 4) as.numeric(args[4]) else 0.6
stopifnot(all(rownames(reads) == rownames(bins)))

param <- data.frame(strength = 1000, e = 0.999999, mu = 0:11, lambda = 20, nu = 2.1,
                    kappa = c(100, 100, 700, 100, rep(25, 8)), m = 0:11, eta = 50000, gamma = 3, S = 1)
multipliers <- 1:6

halfiness <- function(copy, state) {
  # each bin's distance from an integer, on its segment's median copy, as the pipeline scores it
  run <- rleid(state)
  med <- ave(copy, run, FUN = function(x) median(x, na.rm = TRUE))
  h <- -log2(abs(pmin(abs(med - state), 0.499) - 0.5)) - 1
  sum(h / (state + 1), na.rm = TRUE)
}

# The pipeline masks the bins that are not "ideal" (NaN copy) in its own HMMcopy build; with
# Bioconductor's HMMcopy the HMM is run on the ideal bins only, and the others stay uncalled.
segment_at <- function(corrected, ideal, mult) {
  sub <- corrected[ideal]
  sub$chr <- droplevels(sub$chr)
  sub$copy <- sub$cor.gc * mult
  seg <- HMMsegment(sub, param, maxiter = 200, verbose = FALSE)
  list(state = seg$state - 1, copy = sub$copy)
}

states <- matrix(NA_real_, nrow(reads), ncol(reads), dimnames = list(rownames(reads), colnames(reads)))
cells <- data.frame(cell = colnames(reads), multiplier = NA_real_, scaled_halfiness = NA_real_, ok = 0L)
for (j in seq_len(ncol(reads))) {
  d <- data.table(chr = factor(bins$chr, levels = unique(bins$chr)), start = as.integer(bins$start),
                  end = as.integer(bins$end), reads = as.integer(reads[[j]]), gc = as.numeric(bins$gc),
                  map = as.numeric(bins$map))
  res <- try({
    corrected <- correctReadcount(d, mappability = map_cut, verbose = FALSE)
    ideal <- (corrected$ideal %in% TRUE) & is.finite(corrected$cor.gc)
    best <- NULL
    for (v in multipliers) {
      rough <- segment_at(corrected, ideal, v)
      tab <- data.table(state = rough$state, copy = rough$copy)
      meds <- tab[, .(median = median(copy, na.rm = TRUE), n = .N), by = state]
      meds[, fix := state / median]
      fix <- meds[n > 200 & is.finite(fix) & state > 0, mean(fix)]
      true_mult <- if (is.finite(fix) && length(fix)) v * fix else v
      run <- segment_at(corrected, ideal, true_mult)
      pen <- halfiness(run$copy, run$state)
      if (mean(run$state == 1, na.rm = TRUE) > 0.7) pen <- Inf
      if (is.null(best) || pen < best$pen) best <- list(state = run$state, mult = true_mult, pen = pen)
    }
    best$ideal <- ideal
    best
  }, silent = TRUE)
  if (!inherits(res, "try-error") && sum(res$ideal) == length(res$state)) {
    states[res$ideal, j] <- res$state
    cells$multiplier[j] <- res$mult
    cells$scaled_halfiness[j] <- res$pen
    cells$ok[j] <- 1L
  } else if (inherits(res, "try-error")) {
    message(sprintf("cell %s: %s", colnames(reads)[j], conditionMessage(attr(res, "condition"))))
  }
}
cat(sprintf("[hmmcopy_dlp] called %d of %d cells over %d bins\n", sum(cells$ok), ncol(reads), nrow(reads)))
write.csv(states, paste0(prefix, "_state.csv"))
write.csv(cells, paste0(prefix, "_cell.csv"), row.names = FALSE)
