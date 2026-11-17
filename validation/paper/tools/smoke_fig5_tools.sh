#!/bin/bash
# Smoke test of one Fig 5 tool image on synthetic data with a known answer (smoke_fig5_tools.py).
# Everything is written to the job's node-local scratch and deleted with it; the log keeps the
# results and the timings. From the repository root:
#
#   sbatch -A <account> -p cpu -c 4 --mem=16G -t 01:00:00 \
#     --wrap "validation/paper/tools/smoke_fig5_tools.sh integration"      # or mhn, treemhn
#
# ISCC_TOOLS_DIR holds the images and must be set.
set -euo pipefail
TOOL=$1
HERE=$(dirname "$(readlink -f "$0")")
BIN=$HERE/bin
: "${ISCC_TOOLS_DIR:?set ISCC_TOOLS_DIR to the directory of the tool images}"
export ISCC_TOOLS_DIR
export ISCC_WORK_DIR=${SNIC_TMP:?run as a Slurm job}/fig5t-smoke
D=$ISCC_WORK_DIR/data
mkdir -p "$D"
PY=$(command -v python3)
echo "[smoke] $TOOL on $(hostname), ${SLURM_CPUS_PER_TASK:-1} CPUs, image $ISCC_TOOLS_DIR/$TOOL.sif"
timed() { local label=$1 t=$SECONDS; shift; "$@"; echo "[smoke] wall $((SECONDS - t)) s: $label"; }
case "$TOOL" in
  integration)
    timed "make h5ad" "$BIN/python-integration" "$HERE/smoke_fig5_tools.py" h5ad "$D/cells.h5ad" "$D/truth.h5ad"
    for m in none harmony scvi; do
      timed "integrate $m" "$BIN/python-integration" "$HERE/integrate.py" "$D/cells.h5ad" "$D/$m" --method $m
    done
    timed "integrate none (batch-free truth)" "$BIN/python-integration" "$HERE/integrate.py" "$D/truth.h5ad" "$D/truth" --method none
    "$PY" -I "$HERE/smoke_fig5_tools.py" check-integration "$D" ;;
  mhn)
    "$PY" -I "$HERE/smoke_fig5_tools.py" events "$D/events.csv"
    timed "mhn_runner" "$BIN/python-mhn" "$HERE/mhn_runner.py" "$D/events.csv" "$D/mhn"
    "$PY" -I "$HERE/smoke_fig5_tools.py" check-mhn "$D/mhn" ;;
  treemhn)
    "$PY" -I "$HERE/smoke_fig5_tools.py" trees "$D/trees.csv"
    timed "treemhn_runner" "$BIN/rscript-treemhn" "$HERE/treemhn_runner.R" "$D/trees.csv" "$D/treemhn" --n-events 4
    "$PY" -I "$HERE/smoke_fig5_tools.py" check-treemhn "$D/treemhn" ;;
  *) echo "unknown tool $TOOL (integration, mhn, treemhn)" >&2; exit 2 ;;
esac
# the settings files, for the record (the scratch copies go with the job)
for f in "$D"/*.json; do echo "== $(basename "$f")"; cat "$f"; echo; done
