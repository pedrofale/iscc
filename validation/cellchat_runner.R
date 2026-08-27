#!/usr/bin/env Rscript
# Run the REAL CellChat (2.2.0.9001, in the dedicated `iscc-cellchat` conda env) on an iscc Visium
# section, using iscc's OWN invented ligand-receptor database (W0), and write the per-pair spatial
# communication table. This is the recoverability check for W0+W3 (DESIGN_cci_spatial.md): does
# CellChat rank the WIRED pair above the unwired decoys by communication PROBABILITY?
#
# Lives on the far side of the `iscc-cellchat` env so the core `iscc` env never carries R+CellChat.
# The verified round-trip + every silent-failure constraint this obeys is `validation/README_cellchat.md`.
#
# Inputs written by validate_cci.py into <work_dir>:
#   iscc_interaction_input.csv / iscc_complex_input.csv / iscc_cofactor_input.csv /
#   iscc_geneInfo_input.csv  — the four Update-CellChatDB tables (iscc.integrations.write_cci_database)
#   sp_counts.csv   genes x spots (integer counts)
#   sp_coords.csv   spots x 2 (x_um, y_um)   — coordinates already in MICROMETRES (see UNITS below)
#   sp_meta.csv     spots x 1 (group label per spot)
#
# UNITS (a real decision, README §8.3). iscc Visium coordinates are in DEME units; this script is
# handed coordinates already scaled to micrometres by validate_cci.py (deme x DEME_MICRONS). We author
# spatial.factors ratio=1 (coords are um) and COMPUTE scale.distance = 1.02 / min_offdiagonal_distance
# so CellChat's `min(scaled distance) >= 1` constraint holds by construction rather than by luck.
#
# Usage: Rscript cellchat_runner.R <work_dir> <out_csv>
suppressMessages({library(CellChat); library(Matrix)})

args <- commandArgs(trailingOnly = TRUE)
work_dir <- args[1]
out_csv  <- args[2]

# ---- the iscc database ------------------------------------------------------------------------
interaction <- read.csv(file.path(work_dir, "iscc_interaction_input.csv"), row.names = 1,
                        check.names = FALSE, stringsAsFactors = FALSE)
geneInfo    <- read.csv(file.path(work_dir, "iscc_geneInfo_input.csv"), check.names = FALSE,
                        stringsAsFactors = FALSE)
complex_in  <- read.csv(file.path(work_dir, "iscc_complex_input.csv"), check.names = FALSE,
                        stringsAsFactors = FALSE)
cofactor_in <- read.csv(file.path(work_dir, "iscc_cofactor_input.csv"), check.names = FALSE,
                        stringsAsFactors = FALSE)

db.iscc <- updateCellChatDB(db = interaction, gene_info = geneInfo,
                            other_info = list(complex = complex_in, cofactor = cofactor_in))

# THE assertion (README §4.1/§7): every gene the DB references must survive into the analysis, or
# CellChat silently drops the pair. Fail loudly here instead.
expected <- sort(unique(c(interaction$ligand, interaction$receptor)))
present  <- sort(unique(extractGene(db.iscc)))
if (!setequal(present, expected)) {
  stop(sprintf("geneInfo whitelist incomplete: %d expected vs %d in DB (missing: %s)",
               length(expected), length(present),
               paste(head(setdiff(expected, present)), collapse = ",")))
}
cat(sprintf("DB OK: %d interactions over %d genes\n", nrow(interaction), length(present)))

# ---- the Visium section -----------------------------------------------------------------------
sp_counts <- read.csv(file.path(work_dir, "sp_counts.csv"), row.names = 1, check.names = FALSE)
sp_counts <- as.matrix(sp_counts)                       # genes x spots
coords    <- read.csv(file.path(work_dir, "sp_coords.csv"), row.names = 1, check.names = FALSE)
meta      <- read.csv(file.path(work_dir, "sp_meta.csv"),  row.names = 1, check.names = FALSE)

coords <- as.matrix(coords[, 1:2])                      # spots x 2, already in um
meta$group <- factor(meta$group)
meta$samples <- factor("section1")                      # single section; must be a factor (§5)
rownames(meta) <- colnames(sp_counts)

# scale.distance from the geometry: min positive pairwise distance (a robust proxy for spot pitch).
dmat <- as.matrix(dist(coords))
diag(dmat) <- Inf
min_d <- min(dmat)
scale.distance <- 1.02 / min_d                          # guarantees min(scaled) >= 1 (§8.3)
spot_size <- min_d                                      # ~one spot pitch in um
spatial.factors <- data.frame(ratio = 1, tol = spot_size / 2)   # ratio=1: coords are um (§5)
cat(sprintf("geometry: %d spots, min spot distance %.1f um, scale.distance %.4f\n",
            nrow(coords), min_d, scale.distance))

cc <- createCellChat(object = sp_counts, meta = meta, group.by = "group",
                     datatype = "spatial", coordinates = coords,
                     spatial.factors = spatial.factors)
cc@DB <- db.iscc
cc <- normalizeData(cc)
cc <- subsetData(cc)
cc <- identifyOverExpressedGenes(cc)
cc <- identifyOverExpressedInteractions(cc)

# Spatial communication probability. contact/interaction ranges in um, derived from the spot pitch so
# neighbouring spots communicate. `contact.range` is mandatory in spatial mode (§8.2).
cc <- computeCommunProb(cc, type = "triMean", distance.use = TRUE,
                        interaction.range = 4 * min_d, scale.distance = scale.distance,
                        contact.dependent = TRUE, contact.range = 1.5 * min_d, nboot = 100)
cc <- filterCommunication(cc, min.cells = 5)

net <- subsetCommunication(cc)                          # data.frame over OUR pairs
if (is.null(net) || nrow(net) == 0) {
  # write an empty table with the expected columns so the Python side reports "no edges" cleanly
  net <- data.frame(source = character(), target = character(), ligand = character(),
                    receptor = character(), interaction_name = character(),
                    pathway_name = character(), prob = numeric(), pval = numeric())
}
write.csv(net, out_csv, row.names = FALSE)
cat(sprintf("CellChat done: %d communication edges over %d pairs -> %s\n",
            nrow(net), length(unique(net$interaction_name)), out_csv))
