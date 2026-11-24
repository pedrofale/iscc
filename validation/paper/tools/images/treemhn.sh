# TreeMHN (Luo, Kuipers & Beerenwinkel 2023) for Fig 5, Part II: cbg-ethz/TreeMHN at its last commit
# (44f18b5, 2024-01-09; DESCRIPTION 0.1.0), with jsonlite for the runner's settings file.
# Runner: validation/paper/tools/treemhn_runner.R.
#
# Two upstream build problems (README_integration.md, "TreeMHN" install notes):
#   * src/ carries stale object files and a TreeMHN.so from the authors' build; installed over, they
#     give a library without the Rcpp symbols, so they are deleted and the build is clean.
#   * src/Makevars sets CXX_STD = CXX (not a standard R knows) and links -lstdc++fs; the sources use
#     neither std::filesystem nor OpenMP pragmas, so it is replaced by the standard C++17 + OpenMP form.
# TreeMHN's dependencies are installed from conda-forge (install_github does not resolve them there).
# RcppArmadillo is held below 15, whose Armadillo drops API that 2023 code may use.
micromamba install -y -p /opt/conda -c conda-forge r-base=4.4 r-rcpp "r-rcpparmadillo<15" r-matrix \
  r-gtools r-diagrammer r-ggplot2 r-dplyr r-ggm r-reshape2 r-gridextra r-ggpubr r-jsonlite \
  compilers make git
export PATH=/opt/conda/bin:$PATH
COMMIT=44f18b5186b4f3e6ce09d638b140b0ecd73ebdd5
git clone -q https://github.com/cbg-ethz/TreeMHN.git /opt/src/TreeMHN
git -C /opt/src/TreeMHN checkout -q $COMMIT
rm -f /opt/src/TreeMHN/src/*.o /opt/src/TreeMHN/src/*.so /opt/src/TreeMHN/TreeMHN_0.1.0.tar.gz
printf '%s\n' 'CXX_STD = CXX17' 'PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)' \
  'PKG_LIBS = $(SHLIB_OPENMP_CXXFLAGS) $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)' > /opt/src/TreeMHN/src/Makevars
R CMD INSTALL --preclean --clean /opt/src/TreeMHN
rm -rf /opt/src
Rscript -e 'suppressMessages(library(TreeMHN)); stopifnot(exists("learn_MHN"), exists("input_tree_df"))
  v <- sapply(c("TreeMHN", "Rcpp", "RcppArmadillo", "Matrix", "jsonlite"),
              function(p) as.character(packageVersion(p)))
  cat(paste0(names(v), " ", v, "\n"), sep = "", file = "/opt/versions.txt")
  cat("TreeMHN_commit 44f18b5186b4f3e6ce09d638b140b0ecd73ebdd5\n", R.version.string, "\n",
      sep = "", file = "/opt/versions.txt", append = TRUE)'
micromamba clean -a -y
