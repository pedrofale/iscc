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
mode     <- if (length(args) >= 3 && nzchar(args[3])) args[3] else "spatial"   # "spatial" | "rna"
circle_png <- if (length(args) >= 4 && nzchar(args[4])) args[4] else ""

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

# ---- the sample: a Visium SECTION (spatial mode) or DISSOCIATED CELLS (rna mode) ---------------
# CCI inference is overwhelmingly done on dissociated scRNA — CellPhoneDB/CellChat/NATMI/LIANA all
# score cell TYPES from mean L/R expression with NO positions at all. Spatial mode is the newer
# minority. iscc can run both from the SAME tumour and the same planted channel, which is what makes
# the no-position assumption measurable rather than merely criticisable.
if (mode == "rna") {
  counts <- as.matrix(read.csv(file.path(work_dir, "sc_counts.csv"), row.names = 1,
                               check.names = FALSE))            # genes x cells
  meta <- read.csv(file.path(work_dir, "sc_meta.csv"), row.names = 1, check.names = FALSE)
  meta$group <- factor(meta$group)
  meta$samples <- factor("sample1")
  rownames(meta) <- colnames(counts)
  cat(sprintf("RNA mode: %d cells, %d genes, %d groups\n",
              ncol(counts), nrow(counts), nlevels(meta$group)))
  data.norm <- normalizeData(counts)
  cc <- createCellChat(object = data.norm, meta = meta, group.by = "group")
} else {
  sp_counts <- as.matrix(read.csv(file.path(work_dir, "sp_counts.csv"), row.names = 1,
                                  check.names = FALSE))         # genes x spots
  coords <- read.csv(file.path(work_dir, "sp_coords.csv"), row.names = 1, check.names = FALSE)
  meta   <- read.csv(file.path(work_dir, "sp_meta.csv"),  row.names = 1, check.names = FALSE)
  coords <- as.matrix(coords[, 1:2])                    # spots x 2, already in um
  meta$group <- factor(meta$group)
  meta$samples <- factor("section1")                    # single section; must be a factor (§5)
  rownames(meta) <- colnames(sp_counts)

  # scale.distance from the geometry: min positive pairwise distance (a proxy for spot pitch).
  dmat <- as.matrix(dist(coords)); diag(dmat) <- Inf
  min_d <- min(dmat)
  scale.distance <- 1.02 / min_d                        # guarantees min(scaled) >= 1 (§8.3)
  spatial.factors <- data.frame(ratio = 1, tol = min_d / 2)     # ratio=1: coords are um (§5)
  cat(sprintf("geometry: %d spots, min spot distance %.1f um, scale.distance %.4f\n",
              nrow(coords), min_d, scale.distance))
  # `normalizeData` takes the raw COUNT MATRIX (README §3). createCellChat with a plain matrix assumes
  # it is ALREADY normalised (it fills @data, not @data.raw), so normalise FIRST — otherwise a later
  # normalizeData(object) runs colSums on the empty @data.raw and dies on dimensions.
  data.norm <- normalizeData(sp_counts)
  cc <- createCellChat(object = data.norm, meta = meta, group.by = "group",
                       datatype = "spatial", coordinates = coords,
                       spatial.factors = spatial.factors)
}

cc@DB <- db.iscc
cc <- subsetData(cc)
cc <- identifyOverExpressedGenes(cc)
cc <- identifyOverExpressedInteractions(cc)

if (mode == "rna") {
  # No geometry: probability is group-mean ligand x group-mean receptor, proximity ASSUMED.
  cc <- computeCommunProb(cc, type = "triMean", nboot = 100)
} else {
  # Spatial: contact/interaction ranges in um from the spot pitch. contact.range is mandatory (§8.2).
  cc <- computeCommunProb(cc, type = "triMean", distance.use = TRUE,
                          interaction.range = 4 * min_d, scale.distance = scale.distance,
                          contact.dependent = TRUE, contact.range = 1.5 * min_d, nboot = 100)
}
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

# ---- the circle plot: CellChat's OWN aggregated-network view ------------------------------------
# The most recognisable figure in this literature, drawn by CellChat's own plotting code rather than
# a reimplementation. netVisual(..., layout="spatial") is BROKEN upstream in 2.2.0.9001 (it passes an
# idents.use that is not one of its formals), so the aggregate circle view is what we render.
if (nzchar(circle_png)) {
  ok <- try({
    cc <- aggregateNet(cc)
    png(circle_png, width = 1500, height = 750, res = 150)
    par(mfrow = c(1, 2), xpd = TRUE)
    netVisual_circle(cc@net$count, weight.scale = TRUE, label.edge = FALSE,
                     title.name = sprintf("interactions (n) - %s", mode))
    netVisual_circle(cc@net$weight, weight.scale = TRUE, label.edge = FALSE,
                     title.name = sprintf("interaction strength - %s", mode))
    dev.off()
  }, silent = TRUE)
  if (inherits(ok, "try-error")) {
    cat("circle plot failed:", conditionMessage(attr(ok, "condition")), "\n")
  } else {
    cat("circle plot ->", circle_png, "\n")
  }
}
