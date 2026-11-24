# Mutual Hazard Networks (Schill et al. 2020) for Fig 5, Part II: the `mhn` package (spang-lab/LearnMHN)
# 1.2.3, cMHN optimiser. PyPI ships only its source, a Cython extension, so the image carries a C
# compiler; INSTALL_MHN_NO_CUDA builds the CPU code only. mhn needs Python < 3.13 and numpy < 2.
# Runner: validation/paper/tools/mhn_runner.py.
micromamba install -y -p /opt/conda -c conda-forge python=3.12 pip c-compiler
export PATH=/opt/conda/bin:$PATH
INSTALL_MHN_NO_CUDA=1 python -m pip install --no-cache-dir "mhn==1.2.3"
python - <<'PY'
import importlib.metadata as md, sys
with open("/opt/versions.txt", "w") as f:
    for p in ["mhn", "numpy", "scipy", "pandas"]:
        f.write(f"{p} {md.version(p)}\n")
    f.write(f"python {sys.version.split()[0]}\n")
PY
MPLCONFIGDIR=/tmp/mpl python -c "from mhn.optimizers import cMHNOptimizer; cMHNOptimizer(); print('imports ok')"
micromamba clean -a -y
