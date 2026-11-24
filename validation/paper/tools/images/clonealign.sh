# clonealign (GitHub, pinned) on R + reticulate + TensorFlow 2.15 / TensorFlow Probability 0.23,
# the versions of the laptop's iscc-clonealign environment.
micromamba install -y -p /opt/conda -c conda-forge -c bioconda \
  r-base=4.4 python=3.10 pip r-remotes r-reticulate r-tensorflow r-tfruns r-glue r-matrixstats \
  r-jsonlite r-progress r-r.utils r-dplyr r-tidyr r-ggplot2 r-cowplot r-data.table \
  bioconductor-summarizedexperiment bioconductor-singlecellexperiment compilers make git
export PATH=/opt/conda/bin:$PATH
python -m pip install --no-cache-dir "tensorflow-cpu==2.15.0" "tensorflow-probability==0.23.0"
RETICULATE_PYTHON=/opt/conda/bin/python Rscript -e 'options(repos = "https://cloud.r-project.org")
  remotes::install_github("kieranrcampbell/clonealign@0ca0a03050f099e583b9ced1a9b0c7ad83e6de93", upgrade = "never")
  suppressMessages({library(clonealign); library(tensorflow)})'
python - <<'PY'
import tensorflow as tf, tensorflow_probability as tfp, subprocess
v = subprocess.run(["Rscript", "-e", 'cat(as.character(packageVersion("clonealign")))'], capture_output=True, text=True).stdout
open("/opt/versions.txt", "w").write(f"clonealign {v}\ntensorflow {tf.__version__}\ntensorflow_probability {tfp.__version__}\n")
PY
micromamba clean -a -y
