# cardelino (Bioconductor, via bioconda): assigns scRNA cells to clones from somatic-variant allele
# counts given a clone-by-variant matrix (McCarthy et al. 2020). For the SNV supplement of paper 1.
micromamba install -y -p /opt/conda -c conda-forge -c bioconda \
  r-base=4.4 bioconductor-cardelino r-jsonlite r-optparse r-matrix
export PATH=/opt/conda/bin:$PATH
Rscript -e 'suppressMessages(library(cardelino)); cat("cardelino ", as.character(packageVersion("cardelino")), "\n",
  "R ", R.version$major, ".", R.version$minor, "\n", sep = "")' > /opt/versions.txt
cat /opt/versions.txt
micromamba clean -a -y
