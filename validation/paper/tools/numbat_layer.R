#!/usr/bin/env Rscript
# Numbat on a human-gene scRNA dataset written by iscc.data.human.write_scrna, after numbat_inputs.py.
#
# Reads the sparse matrix (gene symbols), <dataset>/numbat/ (df_allele.tsv.gz, gtf.tsv) and the
# "normal" barcodes of <dataset>/clonalscope/celltype.tsv, which make the expression reference
# (aggregate_counts). Writes the common output of this directory (see calls.py): calls.tsv (p_malignant =
# Numbat's p_cnv, malignant above 0.5, clone = clone_opt), cn_arm.tsv (each cell's total copy number per
# arm from the joint posterior's states, 2 where Numbat calls nothing; scale "total_cn"), run.json, and
# Numbat's own files in native/. If Numbat finds no CNV, every cell is written diploid and unassigned.
#
# Usage: Rscript numbat_layer.R <dataset_dir> <out_dir> [n_cores] [max_iter] [min_cells]
suppressMessages({library(numbat); library(Matrix); library(data.table); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
ds <- args[1]; out <- args[2]
cores <- if (length(args) >= 3) as.integer(args[3]) else 1L
max_iter <- if (length(args) >= 4) as.integer(args[4]) else 2L
min_cells <- if (length(args) >= 5) as.integer(args[5]) else 50L
native <- file.path(out, "native")
dir.create(native, recursive = TRUE, showWarnings = FALSE)
t0 <- Sys.time()

mdir <- file.path(ds, "outs", "filtered_feature_bc_matrix")
X <- readMM(file.path(mdir, "matrix.mtx.gz"))
bcs <- fread(file.path(mdir, "barcodes.tsv.gz"), header = FALSE)[[1]]
feat <- fread(file.path(mdir, "features.tsv.gz"), header = FALSE)
rownames(X) <- feat[[2]]; colnames(X) <- bcs
X <- as(X[!duplicated(rownames(X)), ], "dgCMatrix")
gtf <- as.data.frame(fread(file.path(ds, "numbat", "gtf.tsv")))
df_allele <- as.data.frame(fread(file.path(ds, "numbat", "df_allele.tsv.gz")))
df_allele$cell <- as.character(df_allele$cell)
ct <- fread(file.path(ds, "clonalscope", "celltype.tsv"))
normal <- intersect(ct[[1]][ct[[2]] == "normal"], colnames(X))
lambdas_ref <- aggregate_counts(X[, normal, drop = FALSE], data.frame(cell = normal, group = "normal"))
arms <- fread(file.path(ds, "clonalscope", "chrarm.tsv"))

err <- NULL
ok <- tryCatch({
  run_numbat(X, lambdas_ref, df_allele, gtf = gtf, genome = "hg38", out_dir = native,
             min_cells = min_cells, ncores = cores, ncores_nni = cores, max_iter = max_iter,
             t = 1e-5, plot = FALSE, verbose = FALSE)
  TRUE
}, error = function(e) { err <<- conditionMessage(e); cat(sprintf("run_numbat error: %s\n", err)); FALSE })
iters <- as.integer(gsub(".*joint_post_(\\d+)\\.tsv$", "\\1", list.files(native, pattern = "joint_post_\\d+\\.tsv$")))
nb <- if (ok && length(iters)) tryCatch(Numbat$new(native, i = max(iters)), error = function(e) NULL) else NULL

cells <- colnames(X)
calls <- data.table(barcode = cells, malignant = 0L, p_malignant = 0, clone = NA_character_)
cn <- data.table(barcode = rep(cells, each = nrow(arms)), arm = rep(arms$arm, times = length(cells)),
                 value = 2, scale = "total_cn")
status <- if (is.null(err)) "no CNV found" else paste("error:", err)
if (!is.null(nb) && !is.null(nb$clone_post)) {
  cp <- as.data.table(nb$clone_post)
  m <- match(calls$barcode, cp$cell)
  if ("p_cnv" %in% names(cp)) calls$p_malignant <- ifelse(is.na(m), 0, cp$p_cnv[m])
  calls$malignant <- as.integer(calls$p_malignant > 0.5)
  if ("clone_opt" %in% names(cp)) calls$clone <- as.character(cp$clone_opt[m])
  jp <- as.data.table(nb$joint_post)
  state_cn <- c(neu = 2, del = 1, amp = 3, loh = 2, bamp = 4, bdel = 0, "amp+" = 4, "del-" = 0)
  sc <- if ("cnv_state_map" %in% names(jp)) "cnv_state_map" else "cnv_state"
  if (nrow(jp) && all(c("seg_start", "seg_end", "CHROM") %in% names(jp))) {
    jp[, cnv := state_cn[as.character(get(sc))]][is.na(cnv), cnv := 2]
    mid <- (jp$seg_start + jp$seg_end) / 2
    jp$arm <- NA_character_
    for (i in seq_len(nrow(arms))) jp[as.integer(CHROM) == arms$chr[i] & mid >= arms$start[i] & mid < arms$end[i], arm := arms$arm[i]]
    per <- jp[!is.na(arm), .(v = mean(cnv)), by = .(barcode = cell, arm)]
    cn[per, value := i.v, on = .(barcode, arm)]
  }
  status <- sprintf("iteration %d", max(iters))
}
fwrite(calls, file.path(out, "calls.tsv"), sep = "\t", na = "")
fwrite(cn, file.path(out, "cn_arm.tsv"), sep = "\t")
write_json(list(tool = "numbat", version = as.character(packageVersion("numbat")), status = status,
                n_barcodes = length(cells), n_reference = length(normal), max_iter = max_iter,
                min_cells = min_cells, n_cores = cores,
                wall_s = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)),
           file.path(out, "run.json"), auto_unbox = TRUE, pretty = TRUE)
cat("numbat done:", status, ";", sum(calls$malignant), "of", nrow(calls), "called aneuploid\n")
