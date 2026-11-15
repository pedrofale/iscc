# R data simulators for the data-simulator supplement (supp_datasim.py):
#   Splatter 1.34.0 and scDesign3 1.8.0 (Bioconductor, from bioconda), SRTsim 0.99.7 (archived by
#   CRAN on 2025-12-17, so installed from its last release tarball).
# SRTsim's Shiny and 3D-mesh imports (Morpho -> rgl, Rvcg) come from conda-forge where they exist;
# Morpho itself is CRAN-only and is compiled here.
micromamba install -y -p /opt/conda -c conda-forge -c bioconda \
  r-base=4.5 "bioconductor-splatter=1.34.0" "bioconductor-scdesign3=1.8.0" \
  bioconductor-singlecellexperiment bioconductor-s4vectors r-matrix r-jsonlite \
  r-sf r-sp r-concaveman r-spatstat.geom r-spatstat.random r-pdist r-matrixstats r-plotly r-shiny \
  r-viridis r-shinydashboard r-shinybs r-dashboardthemes r-ggplot2 r-ggpubr r-dt r-magrittr r-fnn \
  r-dplyr r-magick r-rgl r-rvcg r-bezier r-rcpparmadillo r-rcppeigen r-doparallel r-foreach \
  r-colorramps r-crayon r-mclust compilers make
export PATH=/opt/conda/bin:$PATH
Rscript -e 'options(repos = "https://cloud.r-project.org", Ncpus = 8)
  install.packages("Morpho", dependencies = c("Depends", "Imports", "LinkingTo"))
  install.packages("https://cran.r-project.org/src/contrib/Archive/SRTsim/SRTsim_0.99.7.tar.gz",
                   repos = NULL, type = "source")
  for (p in c("splatter", "scDesign3", "SRTsim")) suppressMessages(library(p, character.only = TRUE))'
Rscript -e 'v <- sapply(c("splatter", "scDesign3", "SRTsim", "SingleCellExperiment", "Matrix"),
  function(p) as.character(packageVersion(p))); cat(paste0(names(v), " ", v, "\n"), sep = "",
  file = "/opt/versions.txt"); cat(R.version.string, "\n", file = "/opt/versions.txt", append = TRUE)'
micromamba clean -a -y
