#!/usr/bin/env Rscript
# Clonalscope on a human-gene dataset written by iscc.data.human.write_visium or write_scrna, in the
# mode its tutorial gives for data without matched bulk DNA: one first round on chromosome-arm
# segments (samples/P5847/scRNA in the Clonalscope repository).
#
# Reads <dataset>/outs/filtered_feature_bc_matrix/ and <dataset>/clonalscope/ (genes.bed, sizes.txt,
# chrarm.tsv, celltype.tsv). celltype.tsv names the barcodes Clonalscope normalises against
# ("normal"); the rest are clustered. Writes the common output of this directory (see calls.py):
# calls.tsv (clone = Clonalscope's cluster; no malignant call, which Clonalscope makes only with
# bulk DNA), cn_arm.tsv (its per-segment coverage estimates, scale "relative"), run.json, and the
# fitted object in native/Cov_obj.rds.
#
# Usage: Rscript clonalscope.R <dataset_dir> <out_dir> <clonalscope_data_dir> [mincell] [seed] [prior_dir]
#   clonalscope_data_dir holds cyclegenes.rds from the Clonalscope repository's data-raw/.
#
# prior_dir (optional) is a DNA-derived prior, handed over through Clonalscope's own inputs for DNA
# information (RunCovCluster's documentation): states.tsv (arm, state) fills the "states" column of the
# segment table, the copy-number state per segment that Segmentation_bulk gives from matched WGS/WES (1 =
# no change; Clonalscope reads > 1 as gain and < 1 as loss), and U0.tsv (one row per prior cluster, one
# column per arm) is U0, "the prior mean of relative coverage for each cluster". U0 must have one column
# per segment that survives Clonalscope's filtering, so the coverage estimates are made first, exactly as
# RunCovCluster makes them, and RunCovCluster then reads them back (Est_read1 = TRUE) and runs as it
# does without a prior. Cluster k of the output is row k of U0 for k up to nrow(U0); later clusters are
# ones the sampler opened.
suppressMessages({library(Clonalscope); library(Matrix); library(data.table); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
ds <- args[1]; out <- args[2]; data_dir <- args[3]
mincell <- if (length(args) >= 4) as.integer(args[4]) else 20L
seed <- if (length(args) >= 5) as.integer(args[5]) else 200L
prior_dir <- if (length(args) >= 6 && nzchar(args[6])) args[6] else NULL
native <- file.path(out, "native")
dir.create(native, recursive = TRUE, showWarnings = FALSE)
t0 <- Sys.time()

mdir <- file.path(ds, "outs", "filtered_feature_bc_matrix")
mtx <- readMM(file.path(mdir, "matrix.mtx.gz"))
barcodes <- read.table(file.path(mdir, "barcodes.tsv.gz"), stringsAsFactors = FALSE, sep = "\t", header = FALSE)
features <- read.table(file.path(mdir, "features.tsv.gz"), stringsAsFactors = FALSE, sep = "\t", header = FALSE)
cs <- file.path(ds, "clonalscope")
size <- read.table(file.path(cs, "sizes.txt"), stringsAsFactors = FALSE)
bed <- read.table(file.path(cs, "genes.bed"), sep = "\t", header = TRUE)
chrarm <- read.table(file.path(cs, "chrarm.tsv"), sep = "\t", header = TRUE, stringsAsFactors = FALSE)
chrarm <- chrarm[order(as.numeric(chrarm[, 1]), as.numeric(chrarm[, 3])), ]
bin_bed <- chrarm[, -2]
seg_table <- data.frame(chr = bin_bed[, 1], start = as.numeric(bin_bed[, 2]), end = as.numeric(bin_bed[, 3]),
                        states = 1, length = as.numeric(bin_bed[, 3]) - as.numeric(bin_bed[, 2]),
                        mean = 0, var = 0, Var1 = seq_len(nrow(bin_bed)), Freq = 50000,
                        chrr = paste0(bin_bed[, 1], ":", bin_bed[, 2]), stringsAsFactors = FALSE)
celltype <- read.table(file.path(cs, "celltype.tsv"), sep = "\t", header = TRUE, stringsAsFactors = FALSE)
cyclegenes <- readRDS(file.path(data_dir, "cyclegenes.rds"))
clustering_barcodes <- celltype[celltype[, 2] != "normal", 1]

# Segment columns are named "chr<chr>-<start>-<end>", with numbers that may print in scientific
# notation ("chr2-9.6e+07-242193529"); one segment per arm here, matched by chromosome and start.
arm_of_cols <- function(cols) {
  parts <- do.call(rbind, strsplit(sub("^chr", "", cols), "-", fixed = TRUE))
  key <- paste(parts[, 1], as.numeric(parts[, 2]))
  arm_of <- setNames(chrarm$arm, paste(chrarm[, 1], as.numeric(chrarm[, 3])))
  unname(arm_of[key])
}

flt <- FilterFeatures(mtx = mtx, barcodes = barcodes, features = features, cyclegenes = cyclegenes)
U0 <- NULL
Est_read1 <- FALSE
if (!is.null(prior_dir)) {
  st <- read.table(file.path(prior_dir, "states.tsv"), sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  seg_table$states <- st$state[match(chrarm$arm, st$arm)]
  seg_table$states[is.na(seg_table$states)] <- 1
  # RunCovCluster's own first two steps, with its arguments (dense_mat = T, ngene_filter = 150)
  deltas_all <- Clonalscope:::EstRegionCov(mtx = as.matrix(flt$mtx), barcodes = flt$barcodes, features = flt$features,
                             bed = bed, celltype0 = celltype, var_pt = 0.99, var_pt_ctrl = 0.99,
                             include = "all", alpha_source = "all", ctrl_region = NULL,
                             seg_table_filtered = seg_table, size = size,
                             plot_path = file.path(native, "cov_hist.pdf"), breaks = 50)
  saveRDS(deltas_all, file.path(native, "deltas_allseg.rds"))
  cols <- colnames(Clonalscope:::PrepCovMatrix(deltas_all = deltas_all, ngene_filter = 150, prep_mode = "intersect")$df)
  u <- as.matrix(read.table(file.path(prior_dir, "U0.tsv"), sep = "\t", header = TRUE, row.names = 1,
                            check.names = FALSE))
  arms_kept <- arm_of_cols(cols)
  if (any(is.na(arms_kept)) || !all(arms_kept %in% colnames(u))) stop("U0 does not cover the segments kept")
  U0 <- pmin(u[, arms_kept, drop = FALSE], 2)            # est_cap below
  Est_read1 <- TRUE
  cat("clonalscope prior:", nrow(U0), "clusters x", ncol(U0), "segments; states",
      paste(seg_table$states, collapse = " "), "\n")
}
Cov_obj <- RunCovCluster(mtx = flt$mtx, barcodes = flt$barcodes, features = flt$features, bed = bed,
                         celltype0 = celltype, var_pt = 0.99, var_pt_ctrl = 0.99, include = "all",
                         alpha_source = "all", ctrl_region = NULL, seg_table_filtered = seg_table,
                         size = size, dir_path = paste0(native, "/"), breaks = 50, prep_mode = "intersect",
                         est_cap = 2, clust_mode = "all", clustering_barcodes = clustering_barcodes,
                         mincell = mincell, seed = seed, U0 = U0, Est_read1 = Est_read1)
saveRDS(Cov_obj, file.path(native, "Cov_obj.rds"))

fin <- Cov_obj$result_final
df <- fin$df_obj$df                                     # clustered barcodes x segments
calls <- data.table(barcode = rownames(df), malignant = NA_integer_, p_malignant = NA_real_,
                    clone = as.character(fin$result$Zest))
fwrite(calls, file.path(out, "calls.tsv"), sep = "\t", na = "")
key_arm <- arm_of_cols(colnames(df))
hit <- !is.na(key_arm)
cn <- data.table(barcode = rep(rownames(df), times = sum(hit)), arm = rep(key_arm[hit], each = nrow(df)),
                 value = as.vector(as.matrix(df[, hit, drop = FALSE])), scale = "relative")
fwrite(cn, file.path(out, "cn_arm.tsv"), sep = "\t")

write_json(list(tool = "clonalscope", version = as.character(packageVersion("Clonalscope")),
                n_barcodes = nrow(barcodes), n_clustered = length(clustering_barcodes),
                n_clusters = length(unique(calls$clone)), mincell = mincell, seed = seed,
                prior = if (is.null(prior_dir)) NULL else list(rows = rownames(U0), segments = ncol(U0),
                                                              states = seg_table$states),
                wall_s = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)),
           file.path(out, "run.json"), auto_unbox = TRUE, pretty = TRUE)
cat("clonalscope done:", length(unique(calls$clone)), "clusters over", nrow(calls), "barcodes\n")
