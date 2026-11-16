# Batch integration across patients (Fig 5, Part I): Harmony (harmonypy), scVI (scvi-tools on CPU
# torch), scanpy and anndata, and the metrics: scib-metrics for the scib-style normalised iLISI and
# cLISI (Luecken et al. 2022) and the silhouette scores. Runner: validation/paper/tools/integrate.py.
# torch comes from PyTorch's CPU index in the same resolve, so nothing pulls a CUDA build.
micromamba install -y -p /opt/conda -c conda-forge python=3.12 pip
export PATH=/opt/conda/bin:$PATH
python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch==2.14.1+cpu" "scvi-tools==1.5.2" "harmonypy==2.1.0" "scanpy==1.12.5" "anndata==0.13.4" \
  "scib-metrics==0.6.2"
python - <<'PY'
import importlib.metadata as md, sys
pk = ["harmonypy", "scvi-tools", "torch", "lightning", "scanpy", "anndata", "scib-metrics", "jax",
      "numpy", "scipy", "scikit-learn", "pandas", "numba", "pynndescent"]
with open("/opt/versions.txt", "w") as f:
    for p in pk:
        f.write(f"{p} {md.version(p)}\n")
    f.write(f"python {sys.version.split()[0]}\n")
PY
# Import every package once, so that a broken install fails the build rather than the first run.
NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/mpl python -c "import harmonypy, scvi, scanpy, anndata, scib_metrics, torch; print('imports ok')"
micromamba clean -a -y
