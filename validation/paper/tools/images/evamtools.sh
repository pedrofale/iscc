# EvAM-Tools (Diaz-Uriarte & Herrera-Nieves), for CBN in Fig 5, Part II, the first tool to cut. Its
# authors' image rdiaz02/evam-common:4.5.0_3.3.2 (R 4.5.0, evamtools 3.3.2, installed and put through
# R CMD check by their Dockerfile-evam-common) is the base, so this recipe only adds jsonlite, checks
# that CBN runs, and writes the versions:
#
#   sbatch ... --export=ALL,RECIPES=<this dir>,BASE_IMAGE=docker://rdiaz02/evam-common:4.5.0_3.3.2 \
#     build_image.sh evamtools <out_dir>
Rscript -e 'if (!requireNamespace("jsonlite", quietly = TRUE))
    install.packages("jsonlite", repos = "https://cloud.r-project.org")
  suppressMessages(library(evamtools))
  set.seed(1)
  x <- matrix(rbinom(400, 1, 0.3), ncol = 4, dimnames = list(NULL, paste0("E", 0:3)))
  t0 <- Sys.time()
  out <- evam(x, methods = "CBN", cores = 1)
  stopifnot(any(grepl("^CBN", names(out)))); print(names(out)[grepl("^CBN", names(out))])
  cat("CBN on 100 x 4 in", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "s\n")
  v <- sapply(c("evamtools", "jsonlite"), function(p) as.character(packageVersion(p)))
  cat(paste0(names(v), " ", v, "\n"), sep = "", file = "/opt/versions.txt")
  cat("base docker://rdiaz02/evam-common:4.5.0_3.3.2\n", R.version.string, "\n", sep = "",
      file = "/opt/versions.txt", append = TRUE)'
