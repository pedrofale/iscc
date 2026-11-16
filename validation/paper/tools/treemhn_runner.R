#!/usr/bin/env Rscript
# Fit TreeMHN to a cohort's event trees (Fig 5, Parts II and III).
#
# Runs inside the treemhn image (images/treemhn.sh; cbg-ethz/TreeMHN 44f18b5), never in envs/iscc:
#
#   bin/rscript-treemhn treemhn_runner.R <trees.csv> <out_prefix> [--n-events N] [--gamma 0.5]
#       [--lambda-s 1] [--events E0,E1,...] [--seed 0]
#
# Input: the trees as iscc.integrations.progression.to_treemhn_trees writes them: columns Patient_ID,
# Tree_ID, Node_ID, Mutation_ID, Parent_ID (other columns, such as n_cells, are dropped: TreeMHN reads
# topology only). The root is Node_ID 1, Mutation_ID 0 and its own parent; event k is Mutation_ID k + 1.
# A tree with no event (root only) is dropped, as TreeMHN rejects it, and the patients are renumbered.
#
# Settings: TreeMHN's defaults (learn_MHN: gamma = 0.5, the L1 penalty on the off-diagonal;
# lambda_s = 1, the sampling rate; M = 100, iterations = 500, MC_threshold = 500), without the
# stability selection of its demo. --n-events fixes the number of events (default: the largest
# Mutation_ID), so an event no tree carries still gets its row and column. Names default to
# E0..E{n-1}, iscc's event_names().
#
# Output: <out_prefix>.theta.csv, the n x n Theta: entry [i, j], i != j, is the log effect of event j
# on the rate of event i, the diagonal each event's base log rate. And <out_prefix>.json: the
# settings, whether the exact likelihood (MLE) or MCEM was used, the versions and the timings.
#
# Temporary files: R's tempdir() comes from TMPDIR, set by bin/rscript-treemhn to the work directory
# (ISCC_WORK_DIR, else $SNIC_TMP), never /tmp; it is recorded in the JSON.

t0 <- Sys.time()
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: treemhn_runner.R <trees.csv> <out_prefix> [--option value ...]")
in_csv <- args[1]; prefix <- args[2]
opt <- list(`n-events` = NA, gamma = 0.5, `lambda-s` = 1, events = NA, seed = 0)
rest <- args[-(1:2)]
if (length(rest) %% 2 != 0) stop("options come as --name value pairs")
for (i in 2 * seq_len(length(rest) %/% 2) - 1) {
  key <- sub("^--", "", rest[i])
  if (!key %in% names(opt)) stop("unknown option ", rest[i])
  opt[[key]] <- rest[i + 1]
}
gamma <- as.numeric(opt$gamma); lambda_s <- as.numeric(opt$`lambda-s`); seed <- as.integer(opt$seed)

suppressMessages({library(TreeMHN); library(jsonlite)})
if (startsWith(normalizePath(tempdir()), "/tmp") && Sys.getenv("ISCC_WORK_DIR") == "")
  warning("R's tempdir() is under /tmp: set TMPDIR (bin/rscript-treemhn does)")
nthreads <- Sys.getenv("OMP_NUM_THREADS", "1")

df <- read.csv(in_csv)
cols <- c("Patient_ID", "Tree_ID", "Node_ID", "Mutation_ID", "Parent_ID")
miss <- setdiff(cols, colnames(df))
if (length(miss)) stop("the trees lack the columns ", paste(miss, collapse = ", "))
df <- df[, cols]
n <- if (is.na(opt$`n-events`)) max(df$Mutation_ID) else as.integer(opt$`n-events`)
if (max(df$Mutation_ID) > n) stop("a Mutation_ID exceeds --n-events")
events <- if (is.na(opt$events)) paste0("E", seq_len(n) - 1) else strsplit(opt$events, ",")[[1]]
if (length(events) != n) stop("--events names ", length(events), " events, the trees have ", n)

# one tree per (Patient_ID, Tree_ID); drop the root-only ones, then number the patients and the
# trees 1, 2, ... (TreeMHN keys trees by Tree_ID across the cohort and indexes them by position)
key <- paste(df$Patient_ID, df$Tree_ID)
has_edge <- tapply(df$Mutation_ID > 0, key, any)
n_dropped <- sum(!has_edge)
df <- df[key %in% names(has_edge)[has_edge], ]
if (nrow(df) == 0) stop("no tree has an event")
key <- paste(df$Patient_ID, df$Tree_ID)
df$Tree_ID <- match(key, unique(key))
pids <- unique(df$Patient_ID)
df$Patient_ID <- match(df$Patient_ID, pids)
n_trees <- length(unique(df$Tree_ID))

tree_obj <- input_tree_df(n = n, tree_df = df, patients = as.character(seq_along(pids)),
                          mutations = events)
mc <- TreeMHN:::get_MC_flags(tree_obj$N, n, tree_obj$trees, 500)
set.seed(seed)
t_fit <- Sys.time()
Theta <- learn_MHN(tree_obj, gamma = gamma, lambda_s = lambda_s, verbose = FALSE,
                   return_Theta_only = TRUE)
t_fit <- as.numeric(difftime(Sys.time(), t_fit, units = "secs"))
rownames(Theta) <- colnames(Theta) <- events

dir.create(dirname(prefix), showWarnings = FALSE, recursive = TRUE)
write.csv(Theta, paste0(prefix, ".theta.csv"))
vers <- tryCatch({
  v <- readLines("/opt/versions.txt")
  setNames(as.list(sub("^\\S+ ", "", v)), sub(" .*$", "", v))
}, error = function(e) list())
rec <- list(
  input = normalizePath(in_csv), theta = normalizePath(paste0(prefix, ".theta.csv")),
  n_events = n, events = events, n_patients = length(pids), n_trees = n_trees,
  n_dropped_root_only = n_dropped,
  event_patients = as.list(setNames(sapply(seq_len(n), function(k)
    length(unique(df$Patient_ID[df$Mutation_ID == k]))), events)),
  model = "TreeMHN::learn_MHN", gamma = gamma, lambda_s = lambda_s, M = 100, iterations = 500,
  MC_threshold = 500, use_EM = FALSE, stability_selection = FALSE,
  inference = if (any(mc)) "MCEM (some trees exceed MC_threshold subtrees)" else "MLE (exact likelihood)",
  seed = seed, omp_threads = nthreads, r_tempdir = normalizePath(tempdir()),
  convention = paste("Theta[i, j] (i != j): log effect of event j on the rate of event i;",
                     "Theta[i, i]: base log rate of event i"),
  versions = vers, host = Sys.info()[["nodename"]], date = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  timings = list(fit_s = round(t_fit, 2),
                 total_s = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 2)))
write_json(rec, paste0(prefix, ".json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat(sprintf("[treemhn_runner] patients=%d trees=%d events=%d gamma=%.3g %s in %.1f s -> %s.theta.csv\n",
            length(pids), n_trees, n, gamma, rec$inference, rec$timings$total_s, prefix))
