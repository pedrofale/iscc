# R copy-number tools: Numbat (CRAN 1.5.2), CopyKAT (GitHub, pinned), Clonalscope (GitHub, pinned)
# and HMMcopy (Bioconductor). Clonalscope's NAMESPACE imports packages its DESCRIPTION does not
# declare (Seurat, AUCell, BiocFileCache, GSEABase, amap, uwot), so they are installed here too.
# CopyKAT may not be redistributed (its Copyright file): this image is for use on the cluster only
# and is never published.
micromamba install -y -p /opt/conda -c conda-forge -c bioconda \
  r-base=4.4 r-remotes r-matrix r-matrixstats r-data.table r-dplyr r-tidyr r-ggplot2 r-optparse \
  r-glue r-rcpparmadillo r-rcpp r-zoo r-vctrs r-igraph r-catools r-stringr r-purrr r-roptim \
  r-magrittr r-quadprog r-mgcv r-vegan r-fastmap r-r.utils r-rcppparallel r-tidygraph r-ggraph \
  r-logger r-phangorn r-reshape2 r-fastmatch r-memoise r-rhpcblasctl r-digest r-mass r-nlme \
  r-vcfr r-scales r-ape r-plyr r-ggrepel r-tibble r-cachem r-graphlayouts r-ggforce r-jsonlite \
  r-paralleldist r-dlm r-gplots r-rcolorbrewer r-mixtools r-cluster r-mcmcpack r-transport \
  r-pheatmap r-cowplot r-hiddenmarkov r-kernsmooth r-seurat r-amap r-uwot \
  bioconductor-aucell bioconductor-biocfilecache bioconductor-gseabase \
  bioconductor-iranges bioconductor-genomicranges bioconductor-ggtree bioconductor-hmmcopy \
  bioconductor-rtracklayer compilers make git
export PATH=/opt/conda/bin:$PATH
Rscript -e 'options(repos = "https://cloud.r-project.org", Ncpus = 8)
  install.packages("numbat", dependencies = c("Depends", "Imports", "LinkingTo"))
  remotes::install_github("navinlabcode/copykat@01924c0999bb660064da8d3e1561615ac6efb76f", upgrade = "never")
  remotes::install_github("seasoncloud/Clonalscope@07dd0c576b6754d18dda86ec8905d8a9be5e77b0", upgrade = "never")
  for (p in c("numbat", "copykat", "Clonalscope", "HMMcopy")) suppressMessages(library(p, character.only = TRUE))'
Rscript -e 'v <- sapply(c("numbat", "copykat", "Clonalscope", "HMMcopy", "data.table", "Matrix"),
  function(p) as.character(packageVersion(p))); cat(paste0(names(v), " ", v, "\n"), sep = "",
  file = "/opt/versions.txt"); cat(R.version.string, "\n", file = "/opt/versions.txt", append = TRUE)'
micromamba clean -a -y
