#!/usr/bin/env Rscript
# SRTsim fitted to the real Visium breast-cancer section, for the data-simulator supplement
# (supp_datasim.py).
#
# Reads <in>/counts.mtx (genes x in-tissue spots, written by `supp_datasim.py prep-visium`) and
# <in>/loc.csv (spot, x, y in spot-pitch units), and runs SRTsim's reference-based tissue-wise
# simulation as documented: createSRT(), srtsim_fit(sim_scheme = "tissue", marginal = "auto_choose"),
# srtsim_count() at the reference locations (each gene's simulated counts are placed by the rank of its
# reference counts). Writes <out>/srtsim.mtx (genes x spots), <out>/srtsim_loc.csv and <out>/srtsim.json.
#
# Usage: Rscript datasim_srtsim.R <in_dir> <out_dir> [n_cores] [seed]
Sys.setenv(RGL_USE_NULL = "TRUE")
suppressMessages({library(Matrix); library(SRTsim); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
ind <- args[1]; out <- args[2]
cores <- if (length(args) >= 3) as.integer(args[3]) else 1L
seed <- if (length(args) >= 4) as.integer(args[4]) else 1L
counts <- as(readMM(file.path(ind, "counts.mtx")), "CsparseMatrix")
loc <- read.csv(file.path(ind, "loc.csv"))
rownames(loc) <- loc$spot
loc <- data.frame(x = loc$x, y = loc$y, label = "tissue", row.names = loc$spot)
rownames(counts) <- paste0("g", seq_len(nrow(counts)))
colnames(counts) <- rownames(loc)
cat("input", nrow(counts), "genes x", ncol(counts), "spots\n")
set.seed(seed)
t0 <- Sys.time()
s <- createSRT(count_in = counts, loc_in = loc)
s <- srtsim_fit(s, sim_scheme = "tissue", marginal = "auto_choose")
t1 <- Sys.time()
s <- srtsim_count(s, numCores = cores)
new <- s@simCounts
xy <- as.data.frame(s@simcolData)
info <- list(package = paste("SRTsim", packageVersion("SRTsim")), seed = seed,
             settings = "createSRT; srtsim_fit(sim_scheme = tissue, marginal = auto_choose); srtsim_count() defaults",
             fit_seconds = as.numeric(difftime(t1, t0, units = "secs")),
             seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")),
             n_genes = nrow(new), n_spots = ncol(new),
             same_locations = identical(colnames(new), rownames(loc)))
cat("SRTsim done in", info$seconds, "s:", nrow(new), "x", ncol(new), "\n")
writeMM(as(new, "CsparseMatrix"), file.path(out, "srtsim.mtx"))
write.csv(data.frame(spot = colnames(new), x = xy[colnames(new), "x"], y = xy[colnames(new), "y"]),
          file.path(out, "srtsim_loc.csv"), row.names = FALSE, quote = FALSE)
writeLines(toJSON(info, auto_unbox = TRUE, digits = NA), file.path(out, "srtsim.json"))
