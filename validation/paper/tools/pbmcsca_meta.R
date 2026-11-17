#!/usr/bin/env Rscript
# Export the cell metadata of SeuratData's pbmcsca (Ding et al. 2020's PBMC benchmark, the Single Cell
# Portal study SCP424's meta.txt; pbmcsca.SeuratData 3.0.0, CC BY 4.0) as a TSV with a `cell` column
# (the Seurat object's cell names), for batch_fits.py. Runs in the r-cnv image (Seurat):
#
#   apptainer exec -B <dirs> r-cnv.sif Rscript pbmcsca_meta.R <pbmcsca.SeuratData_3.0.0.tar.gz> <scratch> <out.tsv.gz>
#
# The package tarball is unpacked into <scratch> (node-local) and only data/pbmcsca.rda is read.
suppressMessages(library(Seurat))
args <- commandArgs(trailingOnly = TRUE)
tarball <- args[1]; scratch <- args[2]; out <- args[3]
dir.create(scratch, showWarnings = FALSE, recursive = TRUE)
untar(tarball, files = "pbmcsca.SeuratData/data/pbmcsca.rda", exdir = scratch)
e <- new.env()
load(file.path(scratch, "pbmcsca.SeuratData/data/pbmcsca.rda"), envir = e)
obj <- get(ls(e)[1], envir = e)
md <- obj[[]]
md$cell <- rownames(md)
cat("pbmcsca:", nrow(md), "cells; columns", paste(colnames(md), collapse = ", "), "\n")
cat("Seurat", as.character(packageVersion("Seurat")), "\n")
print(table(md$Method, md$Experiment))
v2 <- md$Method %in% c("10x Chromium (v2) A", "10x Chromium (v2) B")
print(table(md$CellType[v2], md$Method[v2], useNA = "ifany"))
print(head(md$cell[v2], 3))
z <- gzfile(out, "w")
write.table(md, z, sep = "\t", quote = FALSE, row.names = FALSE)
close(z)
unlink(file.path(scratch, "pbmcsca.SeuratData"), recursive = TRUE)
cat("wrote", out, "\n")
