#!/usr/bin/env Rscript
# CopyKAT on a human-gene dataset written by iscc.data.human.write_visium or write_scrna.
#
# Reads <dataset>/copykat_counts.tsv.gz (genes x barcodes, Ensembl IDs) and <dataset>/clonalscope/
# chrarm.tsv (the arms), runs copykat() with its documented defaults, and writes the common output
# of this directory (see calls.py): calls.tsv (malignant = "aneuploid"; CopyKAT gives no clones, and
# its hierarchical tree is in native/), cn_arm.tsv (the median of CopyKAT's smoothed log ratios over
# each arm's bins, scale "log2ratio"), run.json, and CopyKAT's own files in native/.
#
# Without a reference CopyKAT takes its own diploid baseline from the data, and on a sample that is
# mostly tumour it can take the tumour as the baseline and invert its calls. With use_reference = 1
# the barcodes <dataset>/clonalscope/celltype.tsv labels "normal" go in as norm.cell.names.
#
# Usage: Rscript copykat.R <dataset_dir> <out_dir> [n_cores] [seed] [use_reference 0|1]
suppressMessages({library(copykat); library(data.table); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
ds <- args[1]; out <- args[2]
cores <- if (length(args) >= 3) as.integer(args[3]) else 1L
seed <- if (length(args) >= 4) as.integer(args[4]) else 0L
use_ref <- length(args) >= 5 && as.integer(args[5]) == 1L
native <- file.path(out, "native")
dir.create(native, recursive = TRUE, showWarnings = FALSE)
t0 <- Sys.time()

dt <- fread(file.path(ds, "copykat_counts.tsv.gz"))
raw <- as.matrix(dt[, -1, with = FALSE]); rownames(raw) <- dt[[1]]
barcodes <- colnames(raw)
# CopyKAT builds data frames from the column names, which turns "ACGT...-1" into "ACGT....1"; map back.
back <- setNames(barcodes, make.names(barcodes))

norm <- ""
if (use_ref) {
  ct <- fread(file.path(ds, "clonalscope", "celltype.tsv"))
  norm <- ct[[1]][ct[[2]] == "normal"]
  norm <- unique(c(norm, make.names(norm)))     # as given, and as CopyKAT may rename them internally
}
set.seed(seed)
owd <- setwd(native)
res <- copykat(rawmat = raw, id.type = "E", ngene.chr = 5, win.size = 25, KS.cut = 0.1,
               sam.name = "iscc", distance = "euclidean", norm.cell.names = norm, output.seg = "FALSE",
               plot.genes = "FALSE", genome = "hg20", n.cores = cores)
setwd(owd)

unmangle <- function(x) ifelse(x %in% barcodes, x, back[x])
pred <- as.data.frame(res$prediction, stringsAsFactors = FALSE)
calls <- data.table(barcode = unmangle(pred$cell.names),
                    malignant = as.integer(grepl("aneuploid", pred$copykat.pred)),
                    p_malignant = NA_real_, clone = NA_character_)
calls[grepl("not.defined", pred$copykat.pred), malignant := NA_integer_]
fwrite(calls, file.path(out, "calls.tsv"), sep = "\t", na = "")

# Arms: CopyKAT's bins carry chromosome (1-23) and position; take each arm's median per barcode.
arms <- fread(file.path(ds, "clonalscope", "chrarm.tsv"))
cna <- as.data.frame(res$CNAmat, check.names = FALSE)
pos <- cna[, 1:3]; vals <- as.matrix(cna[, -(1:3), drop = FALSE])
arm_of <- rep(NA_character_, nrow(pos))
for (i in seq_len(nrow(arms))) {
  hit <- as.integer(pos$chrom) == arms$chr[i] & pos$chrompos >= arms$start[i] & pos$chrompos < arms$end[i]
  arm_of[hit] <- arms$arm[i]
}
keep <- !is.na(arm_of)
med <- apply(vals[keep, , drop = FALSE], 2, function(v) tapply(v, arm_of[keep], median))
cn <- data.table(barcode = unmangle(rep(colnames(med), each = nrow(med))),
                 arm = rep(rownames(med), times = ncol(med)), value = as.vector(med), scale = "log2ratio")
fwrite(cn, file.path(out, "cn_arm.tsv"), sep = "\t")

write_json(list(tool = "copykat", version = as.character(packageVersion("copykat")),
                n_barcodes = length(barcodes), n_genes = nrow(raw), n_cores = cores, seed = seed,
                n_reference = if (use_ref) length(norm) else 0L,
                wall_s = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)),
           file.path(out, "run.json"), auto_unbox = TRUE, pretty = TRUE)
cat("copykat done:", sum(calls$malignant, na.rm = TRUE), "of", nrow(calls), "called aneuploid\n")
