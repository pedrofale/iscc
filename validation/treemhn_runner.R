#!/usr/bin/env Rscript
# TreeMHN runner — executed in the dedicated `iscc-treemhn` env, never loaded by the core env.
#
# Fits TreeMHN (Luo, Kuipers & Beerenwinkel 2023, Nat Commun) to a cohort of per-patient MUTATION
# TREES and writes back the learned Theta. Unlike cross-sectional MHN, TreeMHN reads the trees --
# the order in which events arose within each tumour -- which is the observable iscc's fitness
# epistasis actually leaves a mark on.
#
# Theta[i, j] (i != j) is the multiplicative effect of event j on the RATE of event i. This is NOT
# the same parameter as iscc's planted E (a fitness/selection coefficient), so the benchmark scores
# recovered EDGES, not values.
#
# Usage: Rscript treemhn_runner.R <trees.csv> <out_theta.csv> [gamma]
#   trees.csv columns: Patient_ID, Tree_ID, Node_ID, Mutation_ID, Parent_ID  (Mutation_ID 0 = root)

suppressMessages(library(TreeMHN))

args <- commandArgs(trailingOnly = TRUE)
in_csv <- args[1]; out_csv <- args[2]
gamma <- if (length(args) > 2) as.numeric(args[3]) else 0.5

df <- read.csv(in_csv)
n <- max(df$Mutation_ID)                       # number of real events (root is 0)

tree_df <- df[, c("Patient_ID", "Tree_ID", "Node_ID", "Mutation_ID", "Parent_ID")]
patients <- unique(tree_df$Patient_ID)

tree_obj <- input_tree_df(n = n, tree_df = tree_df,
                          patients = as.character(patients),
                          mutations = paste0("E", seq_len(n) - 1))

# gamma is TreeMHN's L1 penalty on the off-diagonal -- the knob that decides how many edges it is
# willing to report. Left at the package default unless overridden.
Theta <- learn_MHN(tree_obj, gamma = gamma, verbose = FALSE, return_Theta_only = TRUE)

rownames(Theta) <- colnames(Theta) <- paste0("E", seq_len(n) - 1)
write.csv(Theta, out_csv)
cat(sprintf("[treemhn_runner] patients=%d events=%d gamma=%.3f -> %s\n",
            length(patients), n, gamma, out_csv))
