# CINner end to end at a given population size, for the runtime comparison.
#
#   Rscript cinner_runtime.R <CINner repo> <n_cells> <n_sampled> <seed> <out.json>
#
# CINner (dinhngockhanh/CINner) is sourced rather than installed: its simulation core needs only
# ape, ctc, data.table and pbapply, while the package imports some 30 plotting and fitting packages.
# The settings are the package vignette's (copy-number rates, arm and driver selection, 20 founder
# cells), with the population growing logistically from 20 to <n_cells> cells over about 70 cell
# lifespans and sampled at the end, so its growth phase is comparable with the spatial simulators'
# growth to the same size.
#
# One fix is applied to CINner's code, in memory, after sourcing: phase 3 compares
# `standard_time_unit == "day"` where that variable can have length > 1, which stops R; the line
# carries the authors' own note ("Modify here: ... standard_time_unit[1]"), and that is the change.
args <- commandArgs(trailingOnly = TRUE)
repo <- args[1]
n_cells <- as.numeric(args[2])
n_sampled <- as.numeric(args[3])
seed <- as.integer(args[4])
out <- args[5]

suppressMessages({
    library(data.table)
    library(ape)
    library(pbapply)
    library(ctc)
    library(parallel)
})
for (f in list.files(file.path(repo, "R"), pattern = "\\.[rR]$", full.names = TRUE)) {
    tryCatch(source(f), error = function(e) message("skipped ", basename(f), ": ", conditionMessage(e)))
}
step3 <- readLines(file.path(repo, "R", "CINner_step_3.r"))
eval(parse(text = gsub("standard_time_unit ==", "standard_time_unit[1] ==", step3, fixed = TRUE)))
set.seed(seed)

cell_lifespan <- 1
T_0 <- list(0, "day")
n_founders <- 20
t_mid <- log(n_cells / n_founders - 1) / 0.15        # the curve passes through 20 cells at time 0
T_end <- ceiling(t_mid + 40)                          # and is within 0.25% of n_cells at T_end
Table_sample <- data.frame(Sample_ID = "S1", Cell_count = n_sampled, Age_sample = T_end)
vec_time <- 0:T_end
table_population_dynamics <- cbind(vec_time = vec_time,
                                   vec_cell_count = n_cells / (1 + exp(-0.15 * (vec_time - t_mid))))
model_variables <- BUILD_general_variables(
    cell_lifespan = cell_lifespan, T_0 = T_0, T_tau_step = cell_lifespan / 2,
    Table_sample = Table_sample, CN_bin_length = 500000,
    prob_CN_whole_genome_duplication = 0.5e-4, prob_CN_missegregation = 2e-4,
    prob_CN_chrom_arm_missegregation = 2e-4, prob_CN_focal_amplification = 0,
    prob_CN_focal_deletion = 0, prob_neutral_CN_whole_genome_duplication = 0,
    prob_neutral_CN_missegregation = 0, prob_neutral_CN_chrom_arm_missegregation = 0,
    prob_neutral_CN_focal_amplification = 0, prob_neutral_CN_focal_deletion = 0,
    model_CN_focal_amplification_length = "geom", model_CN_focal_deletion_length = "beta",
    prob_CN_focal_amplification_length = 10, prob_CN_focal_deletion_length_shape_1 = 1,
    prob_CN_focal_deletion_length_shape_2 = 7, rate_driver = 1e-3, rate_passenger = 1e-11,
    selection_model = "chrom-arm-and-driver-gene-selection", bound_driver = 3,
    bound_average_ploidy = 4.5, bound_homozygosity = 0,
    table_population_dynamics = table_population_dynamics
)
cn <- model_variables$cn_info
table_arm_selection_rates <- data.frame(
    Arm_ID = c(paste(cn$Chromosome, "p", sep = ""), paste(cn$Chromosome, "q", sep = "")),
    Chromosome = rep(cn$Chromosome, 2),
    Bin_start = c(rep(1, length(cn$Chromosome)), cn$Centromere_location + 1),
    Bin_end = c(cn$Centromere_location, cn$Bin_count),
    s_rate = runif(2 * length(cn$Chromosome), 1 / 1.2, 1.2)
)
table_gene_selection_rates <- data.frame(
    Gene_ID = c("Driver_1", "Driver_2", "Driver_3"), Gene_role = c("TSG", "ONCOGENE", "TSG"),
    s_rate = runif(3, 1, 1.2), Chromosome = c(1, 2, 3), Bin = c(10, 20, 30)
)
model_variables <- BUILD_driver_library(model_variables = model_variables,
                                        table_arm_selection_rates = table_arm_selection_rates,
                                        table_gene_selection_rates = table_gene_selection_rates)
model_variables <- BUILD_initial_population(model_variables = model_variables, cell_count = n_founders,
                                            CN_matrix = BUILD_cn_normal_autosomes(cn), drivers = list())
model_variables <- CHECK_model_variables(model_variables)

wd <- tempfile("cinner_")
dir.create(wd)
old <- setwd(wd)
t0 <- proc.time()[["elapsed"]]
sims <- simulator_full_program(model = model_variables, n_simulations = 1, stage_final = 3,
                               save_simulation = FALSE, report_progress = FALSE,
                               compute_parallel = FALSE, seed = seed)
wall <- proc.time()[["elapsed"]] - t0
setwd(old)
peak_kb <- as.numeric(system(paste("ps -o rss= -p", Sys.getpid()), intern = TRUE))
sampled <- nrow(sims[[1]]$sample$table_cell_clone)
writeLines(sprintf('{"tool": "CINner", "n_cells": %d, "n_sampled": %d, "seed": %d, "wall_s": %.3f, "rss_now_mb": %.1f}',
                   as.integer(n_cells), as.integer(sampled), seed, wall, peak_kb / 1024), out)
