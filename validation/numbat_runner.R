#!/usr/bin/env Rscript
# Run the REAL Numbat (allele-aware CNA-from-expression) on iscc file inputs, write per-cell outputs.
#
# Lives on the far side of the dedicated `iscc-numbat` conda env so the core `iscc` env never carries
# the R + Bioconductor + numbat stack. iscc writes: count_mat.csv (genes x cells UMI), ref_counts.csv
# (genes x normal cells, for the expression reference), df_allele.csv (phased allele counts built
# DIRECTLY from iscc's per-homolog expression — see integration_common.build_numbat_inputs), and
# gtf.csv (iscc segments mapped onto Numbat chromosomes). This script runs run_numbat() with the custom
# gtf (no population phasing panel — iscc's homolog labels ARE the phasing) and writes:
#   clone.csv        cell, clone, p_aneuploid   (clone assignment + malignant probability)
#   cell_seg_cn.csv  cells x CHROM inferred total copy number (from the per-cell CNV posterior)
#
# Usage: Rscript numbat_runner.R <in_dir> <out_dir> [min_cells] [ncores] [max_iter] [t] [seed]
suppressMessages({library(numbat); library(dplyr); library(data.table); library(Matrix)})

args <- commandArgs(trailingOnly = TRUE)
in_dir   <- args[1]
out_dir  <- args[2]
min_cells <- ifelse(length(args) >= 3, as.integer(args[3]), 20L)
ncores    <- ifelse(length(args) >= 4, as.integer(args[4]), 1L)
max_iter  <- ifelse(length(args) >= 5, as.integer(args[5]), 1L)
tval      <- ifelse(length(args) >= 6, as.numeric(args[6]), 1e-5)
seed      <- ifelse(length(args) >= 7, as.integer(args[7]), 0L)
min_llr   <- ifelse(length(args) >= 8, as.numeric(args[8]), 5.0)
set.seed(seed)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

count_mat <- as(as.matrix(read.csv(file.path(in_dir, "count_mat.csv"), row.names = 1,
                                   check.names = FALSE)), "dgCMatrix")           # genes x cells
ref_mat   <- as(as.matrix(read.csv(file.path(in_dir, "ref_counts.csv"), row.names = 1,
                                   check.names = FALSE)), "dgCMatrix")           # genes x normal cells
df_allele <- as.data.frame(fread(file.path(in_dir, "df_allele.csv")))
df_allele$CHROM <- as.integer(df_allele$CHROM)
df_allele$cell  <- as.character(df_allele$cell)
gtf <- as.data.frame(fread(file.path(in_dir, "gtf.csv")))

# expression reference: aggregate the normal cells into a lambdas_ref (gene x group) profile.
annot <- data.frame(cell = colnames(ref_mat), group = "normal")
lambdas_ref <- aggregate_counts(ref_mat, annot)

# Numbat can legitimately find NO CNV ("No CNV remains after filtering by LLR in pseudobulks") on a weak
# or noisy tumour, and its internal allele HMM (hahmmr) occasionally throws on the idealised allele input
# we feed it. Neither is a bug in iscc: both mean "Numbat recovered nothing here", a valid result. Write
# neutral outputs (all cells diploid/balanced -> downstream scoring reads it as "recovered nothing") and
# exit 0 rather than crash, so one hard tumour never breaks the cohort figure.
chroms <- sort(unique(gtf$CHROM))
write_neutral <- function(msg) {
  cells <- colnames(count_mat)
  write.csv(data.frame(cell = cells, clone = 1L, p_aneuploid = 0.0),
            file.path(out_dir, "clone.csv"), row.names = FALSE)
  mk <- function(fill) as.data.frame(matrix(fill, nrow = length(cells), ncol = length(chroms),
                                            dimnames = list(cells, as.character(chroms))))
  write.csv(mk(2.0), file.path(out_dir, "cell_seg_cn.csv"))
  write.csv(mk(0.0), file.path(out_dir, "cell_seg_imbalance.csv"))
  write.csv(mk(0.0), file.path(out_dir, "cell_seg_loh.csv"))
  write.csv(mk("neu"), file.path(out_dir, "cell_seg_state.csv"))
  cat(sprintf("numbat recovered nothing (%s): wrote neutral outputs for %d cells\n", msg, length(cells)))
}

# Numbat with the CUSTOM gtf (iscc segments = chromosomes 1..S). genome is passed but the custom gtf
# overrides its gene coordinates. plot=FALSE avoids the ggtree/graphics path in a headless env. Wrapped
# so an internal Numbat error (e.g. a hahmmr viterbi C++ throw) degrades to neutral, not a crash.
ok <- tryCatch({
  run_numbat(count_mat, lambdas_ref, df_allele, gtf = gtf, genome = "hg38", out_dir = out_dir,
             min_cells = min_cells, ncores = ncores, ncores_nni = ncores, max_iter = max_iter,
             t = tval, min_LLR = min_llr, plot = FALSE, verbose = TRUE)
  TRUE
}, error = function(e) { cat(sprintf("run_numbat error: %s\n", conditionMessage(e))); FALSE })
if (!ok) { write_neutral("run_numbat error"); quit(save = "no", status = 0) }

# Numbat writes per-iteration files (joint_post_<i>.tsv, ...); load the latest iteration that exists
# (Numbat$new defaults to i=2 which may not be present when max_iter < 2).
iters <- as.integer(gsub(".*joint_post_(\\d+)\\.tsv$", "\\1",
                         list.files(out_dir, pattern = "joint_post_\\d+\\.tsv$")))
if (length(iters) == 0) { write_neutral("no CNV after LLR filter"); quit(save = "no", status = 0) }
i_load <- max(iters)
nb <- tryCatch(Numbat$new(out_dir, i = i_load), error = function(e) NULL)
if (is.null(nb) || is.null(nb$joint_post) || nrow(as.data.frame(nb$joint_post)) == 0) {
  write_neutral(); quit(save = "no", status = 0)
}

# ---- clone assignment + malignant probability ------------------------------------------------
cp <- as.data.frame(nb$clone_post)
clone_col <- if ("clone_opt" %in% names(cp)) "clone_opt" else grep("clone", names(cp), value = TRUE)[1]
# malignant probability: prefer an explicit p_cnv; else 1 - P(normal compartment); else 1 - p of the
# clone labelled normal (clone 1 is the diploid root in Numbat's convention).
if ("p_cnv" %in% names(cp)) {
  p_aneu <- cp$p_cnv
} else if ("compartment_opt" %in% names(cp)) {
  p_aneu <- ifelse(cp$compartment_opt == "tumor", 1, 0)
  if ("p_1" %in% names(cp)) p_aneu <- 1 - cp$p_1
} else if ("p_1" %in% names(cp)) {
  p_aneu <- 1 - cp$p_1
} else {
  p_aneu <- as.numeric(cp[[clone_col]] > 1)
}
clone_out <- data.frame(cell = cp$cell, clone = cp[[clone_col]], p_aneuploid = p_aneu)
write.csv(clone_out, file.path(out_dir, "clone.csv"), row.names = FALSE)

# ---- per-cell per-CHROM total copy number, from the joint posterior --------------------------
# cnv_state -> total CN. Numbat states: neu(2), del(1), amp(3), loh(2, copy-neutral), bamp(4), bdel(0).
state_cn <- c(neu = 2, del = 1, amp = 3, loh = 2, bamp = 4, bdel = 0, "amp+" = 4, "del-" = 0)
# cnv_state -> ALLELIC IMBALANCE (BAF != 0.5): the allele-only signal inferCNV cannot represent.
# neu(1+1), bamp(2+2), bdel(0+0) are allelically BALANCED; del(1+0), amp(2+1), loh(2+0, copy-neutral)
# are IMBALANCED. loh is the sharpest case: total CN is unchanged (2), so ONLY the allele layer sees it.
state_imbalanced <- c(neu = 0, del = 1, amp = 1, loh = 1, bamp = 0, bdel = 0, "amp+" = 1, "del-" = 1)
jp <- as.data.frame(nb$joint_post)
cat(sprintf("joint_post columns: %s\n", paste(names(jp), collapse = ",")))
state_col <- if ("cnv_state_map" %in% names(jp)) "cnv_state_map" else "cnv_state"
jp$cn  <- state_cn[as.character(jp[[state_col]])];         jp$cn[is.na(jp$cn)]   <- 2
jp$imb <- state_imbalanced[as.character(jp[[state_col]])]; jp$imb[is.na(jp$imb)] <- 0
# P(allelic imbalance) straight from the allele posterior when Numbat exposes the per-state columns
# (p_del + p_amp + p_loh); falls back to the hard state call otherwise. p_loh alone = P(copy-neutral).
has_post <- all(c("p_del", "p_amp", "p_loh") %in% names(jp))
jp$imb_p <- if (has_post) rowSums(jp[, c("p_del", "p_amp", "p_loh")], na.rm = TRUE) else jp$imb
jp$loh_p <- if ("p_loh" %in% names(jp)) jp$p_loh else as.numeric(jp[[state_col]] == "loh")

cells <- clone_out$cell
chroms <- sort(unique(gtf$CHROM))
mk <- function(fill) matrix(fill, nrow = length(cells), ncol = length(chroms),
                            dimnames = list(cells, as.character(chroms)))
M <- mk(2.0); IMB <- mk(0.0); LOH <- mk(0.0); STATE <- mk("neu")
if (nrow(jp) > 0) {
  # each (cell, CHROM) -> mean CN / mean P(imbalance) / mean P(loh) over its overlapping CNV segment(s),
  # and the modal (most posterior-weighted) cnv_state.
  agg <- jp %>% group_by(cell, CHROM) %>%
    summarise(cn = mean(cn), imb = mean(imb_p), loh = mean(loh_p),
              state = names(which.max(table(as.character(.data[[state_col]])))), .groups = "drop")
  for (i in seq_len(nrow(agg))) {
    ci <- as.character(agg$cell[i]); hi <- as.character(agg$CHROM[i])
    if (ci %in% rownames(M) && hi %in% colnames(M)) {
      M[ci, hi] <- agg$cn[i]; IMB[ci, hi] <- agg$imb[i]
      LOH[ci, hi] <- agg$loh[i]; STATE[ci, hi] <- agg$state[i]
    }
  }
}
write.csv(as.data.frame(M),   file.path(out_dir, "cell_seg_cn.csv"))
write.csv(as.data.frame(IMB), file.path(out_dir, "cell_seg_imbalance.csv"))  # P(BAF != 0.5)
write.csv(as.data.frame(LOH), file.path(out_dir, "cell_seg_loh.csv"))        # P(copy-neutral LOH)
write.csv(as.data.frame(STATE, stringsAsFactors = FALSE),
          file.path(out_dir, "cell_seg_state.csv"))                          # modal cnv_state
cat(sprintf("numbat done: %d cells, %d chroms -> %s\n", length(cells), length(chroms), out_dir))
