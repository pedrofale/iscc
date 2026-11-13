#!/bin/bash
# Grow registry runs as a Slurm job array, one tumour per array element.
#
#   RUNS=R1,R3 ISCC_PAPER_TIER=full ISCC_PAPER_OUT=/path/to/storage/paper \
#   ISCC_ENV=/path/to/conda/env SLURM_ACCOUNT=<account> validation/paper/slurm_registry.sh
#
# Submits `registry.py task $RUNS $SLURM_ARRAY_TASK_ID` for every task of RUNS. Tasks already grown
# are skipped, so a failed array can be resubmitted as it is. Afterwards, `registry.py manifest
# $RUNS` rebuilds each run's manifest from the per-task records.
set -euo pipefail
: "${RUNS:?set RUNS, e.g. R1,R3}" "${ISCC_ENV:?set ISCC_ENV to the conda environment}"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT}"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export ISCC_PAPER_TIER=${ISCC_PAPER_TIER:-preview}
N=$(PYTHONPATH="$REPO/src" "$ISCC_ENV/bin/python" "$HERE/registry.py" count "$RUNS")
LOGS=${LOGS:-${ISCC_PAPER_OUT:-$REPO/analysis_data/paper}/logs}
mkdir -p "$LOGS"
sbatch --parsable -A "$SLURM_ACCOUNT" -p "${PARTITION:-cpu}" -J "iscc-${RUNS//,/-}" \
  -n 1 -c 1 --mem="${MEM:-16G}" -t "${TIME:-08:00:00}" --array="0-$((N - 1))%${MAX_PARALLEL:-200}" \
  -o "$LOGS/%x-%A_%a.out" --export=ALL,RUNS="$RUNS",REPO="$REPO" \
  --wrap "PYTHONPATH=\$REPO/src $ISCC_ENV/bin/python \$REPO/validation/paper/registry.py task \$RUNS \$SLURM_ARRAY_TASK_ID"
