#!/usr/bin/env Rscript
# Fit CBN to a cohort's events through EvAM-Tools (Fig 5, Part II; the first tool to cut).
#
# Runs inside the evamtools image (images/evamtools.sh; evamtools 3.3.2), never in envs/iscc:
#
#   bin/rscript-evamtools cbn_runner.R <events.csv> <out_prefix> [--seed 0]
#
# Input: a patients x events 0/1 CSV with a header of event names and the patient ids in its first
# column, as iscc.integrations.progression.to_mhn_matrix(...).to_csv(path) writes it.
# Model: evam(x, methods = "CBN", cores = 1), EvAM-Tools' defaults otherwise.
# Output: <out_prefix>.json with the fitted CBN model exactly as evam returns it (out$CBN_model, as
# records, whatever its columns), the names of every element evam returned, the settings, the
# versions and the timings. A failed fit writes the error message instead, so a collector can
# record it.
# Temporary files: R's tempdir() comes from TMPDIR, set by bin/rscript-evamtools to the work
# directory (ISCC_WORK_DIR, else $SNIC_TMP), never /tmp.

t0 <- Sys.time()
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: cbn_runner.R <events.csv> <out_prefix> [--seed N]")
in_csv <- args[1]; prefix <- args[2]
seed <- 0L
if (length(args) >= 4 && args[3] == "--seed") seed <- as.integer(args[4])
suppressMessages({library(evamtools); library(jsonlite)})

df <- read.csv(in_csv, row.names = 1, check.names = FALSE)
x <- as.matrix(df)
storage.mode(x) <- "integer"
if (!all(x %in% c(0L, 1L))) stop("the event matrix must hold only 0 and 1")
vers <- tryCatch({
  v <- readLines("/opt/versions.txt")
  setNames(as.list(sub("^\\S+ ", "", v)), sub(" .*$", "", v))
}, error = function(e) list())
set.seed(seed)
t_fit <- Sys.time()
res <- tryCatch(evam(x, methods = "CBN", cores = 1), error = function(e) e)
t_fit <- as.numeric(difftime(Sys.time(), t_fit, units = "secs"))
rec <- list(input = normalizePath(in_csv), n_patients = nrow(x), n_events = ncol(x),
            events = colnames(x), event_frequencies = as.list(setNames(colMeans(x), colnames(x))),
            model = "evamtools::evam(methods = 'CBN', cores = 1)", seed = seed,
            versions = vers, host = Sys.info()[["nodename"]],
            date = format(Sys.time(), "%Y-%m-%d %H:%M:%S"), r_tempdir = normalizePath(tempdir()))
if (inherits(res, "error")) {
  rec$error <- conditionMessage(res)
} else {
  rec$returned <- names(res)
  rec$cbn_model <- res$CBN_model
}
rec$timings <- list(fit_s = round(t_fit, 2),
                    total_s = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 2))
dir.create(dirname(prefix), showWarnings = FALSE, recursive = TRUE)
write_json(rec, paste0(prefix, ".json"), auto_unbox = TRUE, pretty = TRUE, digits = NA,
           dataframe = "rows", null = "null", na = "null")
cat(sprintf("[cbn_runner] patients=%d events=%d %s in %.1f s -> %s.json\n", nrow(x), ncol(x),
            if (is.null(rec$error)) "ok" else paste("error:", rec$error), rec$timings$total_s, prefix))
