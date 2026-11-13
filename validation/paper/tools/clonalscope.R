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
# Usage: Rscript clonalscope.R <dataset_dir> <out_dir> <clonalscope_data_dir> [mincell] [seed]
#   clonalscope_data_dir holds cyclegenes.rds from the Clonalscope repository's data-raw/.
suppressMessages({library(Clonalscope); library(Matrix); library(data.table); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
ds <- args[1]; out <- args[2]; data_dir <- args[3]
mincell <- if (length(args) >= 4) as.integer(args[4]) else 20L
seed <- if (length(args) >= 5) as.integer(args[5]) else 200L
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

flt <- FilterFeatures(mtx = mtx, barcodes = barcodes, features = features, cyclegenes = cyclegenes)
Cov_obj <- RunCovCluster(mtx = flt$mtx, barcodes = flt$barcodes, features = flt$features, bed = bed,
                         celltype0 = celltype, var_pt = 0.99, var_pt_ctrl = 0.99, include = "all",
                         alpha_source = "all", ctrl_region = NULL, seg_table_filtered = seg_table,
                         size = size, dir_path = paste0(native, "/"), breaks = 50, prep_mode = "intersect",
                         est_cap = 2, clust_mode = "all", clustering_barcodes = clustering_barcodes,
                         mincell = mincell, seed = seed)
saveRDS(Cov_obj, file.path(native, "Cov_obj.rds"))

fin <- Cov_obj$result_final
df <- fin$df_obj$df                                     # clustered barcodes x segments
calls <- data.table(barcode = rownames(df), malignant = NA_integer_, p_malignant = NA_real_,
                    clone = as.character(fin$result$Zest))
fwrite(calls, file.path(out, "calls.tsv"), sep = "\t", na = "")
# Segment columns are named "chr<chr>-<start>-<end>", with numbers that may print in scientific
# notation ("chr2-9.6e+07-242193529"); one segment per arm here, matched by chromosome and start.
parts <- do.call(rbind, strsplit(sub("^chr", "", colnames(df)), "-", fixed = TRUE))
key <- paste(parts[, 1], as.numeric(parts[, 2]))
arm_of <- setNames(chrarm$arm, paste(chrarm[, 1], as.numeric(chrarm[, 3])))
hit <- key %in% names(arm_of)
cn <- data.table(barcode = rep(rownames(df), times = sum(hit)), arm = rep(arm_of[key[hit]], each = nrow(df)),
                 value = as.vector(as.matrix(df[, hit, drop = FALSE])), scale = "relative")
fwrite(cn, file.path(out, "cn_arm.tsv"), sep = "\t")

write_json(list(tool = "clonalscope", version = as.character(packageVersion("Clonalscope")),
                n_barcodes = nrow(barcodes), n_clustered = length(clustering_barcodes),
                n_clusters = length(unique(calls$clone)), mincell = mincell, seed = seed,
                wall_s = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)),
           file.path(out, "run.json"), auto_unbox = TRUE, pretty = TRUE)
cat("clonalscope done:", length(unique(calls$clone)), "clusters over", nrow(calls), "barcodes\n")
