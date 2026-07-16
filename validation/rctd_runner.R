#!/usr/bin/env Rscript
# Run the REAL RCTD (spacexr) on an iscc reference + Visium section, write per-spot proportions.
#
# Lives on the far side of the dedicated `iscc-rctd` conda env so the core `iscc` env never carries
# the R + spacexr stack. The deconvolution validation writes CSVs (reference counts genes x cells +
# per-cell labels, spatial counts genes x spots + spot coords); this script builds an RCTD Reference
# and SpatialRNA, runs RCTD, and writes the normalised per-spot cell-type weights to CSV.
#
# Usage:  Rscript rctd_runner.R <work_dir> <out_csv> [mode=full] [seed]
suppressMessages({library(spacexr); library(Matrix)})

args <- commandArgs(trailingOnly = TRUE)
work_dir <- args[1]
out_csv  <- args[2]
mode     <- ifelse(length(args) >= 3, args[3], "full")
seed     <- ifelse(length(args) >= 4, as.integer(args[4]), 0L)
set.seed(seed)

read_counts <- function(path) {
  m <- read.csv(path, row.names = 1, check.names = FALSE)
  as(as.matrix(m), "dgCMatrix")
}

ref_counts <- read_counts(file.path(work_dir, "ref_counts.csv"))     # genes x cells
labels     <- read.csv(file.path(work_dir, "ref_labels.csv"), row.names = 1, check.names = FALSE)
sp_counts  <- read_counts(file.path(work_dir, "sp_counts.csv"))       # genes x spots
coords     <- read.csv(file.path(work_dir, "sp_coords.csv"), row.names = 1, check.names = FALSE)

cell_types <- factor(labels[colnames(ref_counts), 1])
names(cell_types) <- colnames(ref_counts)

# RCTD needs >= 25 cells per type by default; drop types below that so the run does not error, and
# require a modest min UMI. nUMI = per-cell / per-spot totals.
min_per_type <- 25
keep_types <- names(which(table(cell_types) >= min_per_type))
keep_cells <- names(cell_types)[cell_types %in% keep_types]
ref_counts <- ref_counts[, keep_cells]
cell_types <- droplevels(cell_types[keep_cells])

reference <- Reference(ref_counts, cell_types, nUMI = colSums(ref_counts), min_UMI = 1)
coords_df <- data.frame(x = coords$x, y = coords$y)
rownames(coords_df) <- rownames(coords)
puck <- SpatialRNA(coords_df, sp_counts, nUMI = colSums(sp_counts), require_int = TRUE)

myRCTD <- create.RCTD(puck, reference, max_cores = 1, CELL_MIN_INSTANCE = min_per_type,
                      UMI_min = 1, counts_MIN = 1, test_mode = FALSE)
myRCTD <- run.RCTD(myRCTD, doublet_mode = mode)

# full mode -> results$weights (spots x cell types); normalise rows to proportions.
if (mode == "full") {
  w <- as.matrix(myRCTD@results$weights)
} else {
  w <- as.matrix(myRCTD@results$weights)
}
w <- sweep(w, 1, pmax(rowSums(w), 1e-9), "/")
w <- as.data.frame(w)
rownames(w) <- rownames(coords)
write.csv(w, out_csv)
cat(sprintf("RCTD done: %d spots x %d cell types -> %s\n", nrow(w), ncol(w), out_csv))
