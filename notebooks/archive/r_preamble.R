# Shared preamble for the R analysis notebooks. Source this FIRST in every one of them:
#
#     source("r_preamble.R")
#
# WHY IT IS NOT OPTIONAL. Several of these packages reach Python through reticulate — clonealign's
# backend is TensorFlow, Numbat pulls scipy — and reticulate does NOT default to the R installation's
# own environment. Left alone it searches a cached uv interpreter and reports things like
#     Error: Valid installation of TensorFlow not found.
# even though the package is installed correctly in this env. Pointing RETICULATE_PYTHON at the env's
# own python is what makes the tool start. (Omitting this cost a debugging cycle on 2026-08-26.)
local({
  env_py <- file.path(dirname(dirname(R.home())), "bin", "python")
  if (file.exists(env_py)) Sys.setenv(RETICULATE_PYTHON = env_py)
})
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")   # quiet TensorFlow's startup chatter in notebook output

# Where the pre-generated datasets live, relative to notebooks/. These notebooks NEVER simulate:
# `python validation/make_analysis_data.py` writes the tables, and the notebook only loads them.
ANALYSIS_DATA <- normalizePath(file.path("..", "analysis_data"), mustWork = FALSE)

analysis_dir <- function(name) {
  d <- file.path(ANALYSIS_DATA, name)
  if (!dir.exists(d)) {
    stop(sprintf(paste0("dataset '%s' not found at %s\n",
                        "  Generate it first:  python validation/make_analysis_data.py --only %s"),
                 name, d, name), call. = FALSE)
  }
  d
}
