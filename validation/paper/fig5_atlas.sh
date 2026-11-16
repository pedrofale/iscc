#!/bin/bash
# Submit one step of Fig 5's atlas cohort (fig5_atlas.py) to Slurm, one summary file per array task.
#
#   STEP=pilot ISCC_ENV=<conda env> SLURM_ACCOUNT=<account> ISCC_FIG5_OUT=<summary dir> \
#     [ISCC_TOOLS_DIR=<images>] [AFTER=<job id>] validation/paper/fig5_atlas.sh
#
# STEP is
#   pilot          an array, one tumour per task (arm x grid x patient), 1 CPU each; SIZES=s0,...,s6
#                  grows one grid at those sizes instead of the registered grid (a confirming round;
#                  EXTRA="--grid-name r2" names it);
#   pilot-collect  one small job;
#   grow           an array, one cohort patient per task; needs SIZES=s0,...,s6 (E0..E6);
#   integrate      one job per batch level: LEVEL=10x|strong|intermediate (16 CPUs: two scVI runs of
#                  6 threads each beside four single-threaded none/Harmony runs);
#   collect        one job, CPUS fits at a time.
# The registered field needs ISCC_PAPER_TIER=full (set here for pilot and grow). EFFECT_SCALE passes
# --effect-scale; N_PATIENTS shrinks the cohort (smoke tests); EXTRA adds options. AFTER makes the job
# wait for another (afterany). Logs go to $ISCC_FIG5_OUT/logs. Never -N 1.
set -euo pipefail
: "${STEP:?set STEP}" "${ISCC_ENV:?set ISCC_ENV}" "${SLURM_ACCOUNT:?set SLURM_ACCOUNT}"
: "${ISCC_FIG5_OUT:?set ISCC_FIG5_OUT}"
HERE=$(cd "$(dirname "$0")" && pwd)
export REPO=$(cd "$HERE/../.." && pwd) ISCC_FIG5_OUT
export ISCC_TOOLS_DIR=${ISCC_TOOLS_DIR:-/nobackup/proj/flash/naiss2025-22-1610/personal/pedrof/iscc/tools}
LOGS=$ISCC_FIG5_OUT/logs
mkdir -p "$LOGS"
PY="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=\$REPO/src $ISCC_ENV/bin/python \$REPO/validation/paper/fig5_atlas.py"
NP=${N_PATIENTS:+--n-patients $N_PATIENTS}
ES=${EFFECT_SCALE:+--effect-scale $EFFECT_SCALE}
DEP=${AFTER:+--dependency=afterany:$AFTER}
SB=(sbatch --parsable -A "$SLURM_ACCOUNT" -p "${PARTITION:-cpu}" -n 1 --export=ALL ${DEP:-})
case "$STEP" in
  pilot|grow)
    export ISCC_PAPER_TIER=full
    [ "$STEP" = grow ] && : "${SIZES:?set SIZES=s0,...,s6 for grow}"
    SZ=${SIZES:+--sizes $SIZES}
    N=$(PYTHONPATH="$REPO/src" "$ISCC_ENV/bin/python" "$HERE/fig5_atlas.py" count "$STEP" ${NP:-} ${SZ:-})
    "${SB[@]}" -J "fig5a-$STEP" -c 1 --mem="${MEM:-8G}" -t "${TIME:-01:00:00}" \
      --array="0-$((N - 1))%${MAX_PARALLEL:-100}" -o "$LOGS/%x-%A_%a.out" \
      --wrap "export TMPDIR=\$SNIC_TMP; $PY $STEP \$SLURM_ARRAY_TASK_ID ${NP:-} ${ES:-} ${SZ:-} ${EXTRA:-}" ;;
  pilot-collect)
    "${SB[@]}" -J "fig5a-$STEP" -c 1 --mem="${MEM:-4G}" -t "${TIME:-00:30:00}" -o "$LOGS/%x-%j.out" \
      --wrap "export TMPDIR=\$SNIC_TMP; $PY $STEP ${EXTRA:-}" ;;
  integrate)
    : "${LEVEL:?set LEVEL=10x|strong|intermediate}"
    "${SB[@]}" -J "fig5a-int-$LEVEL" -c "${CPUS:-16}" --mem="${MEM:-96G}" -t "${TIME:-08:00:00}" \
      -o "$LOGS/%x-%j.out" \
      --wrap "export TMPDIR=\$SNIC_TMP; $PY integrate --level $LEVEL ${NP:-} ${EXTRA:-}" ;;
  collect)
    "${SB[@]}" -J "fig5a-$STEP" -c "${CPUS:-4}" --mem="${MEM:-16G}" -t "${TIME:-02:00:00}" \
      -o "$LOGS/%x-%j.out" \
      --wrap "export TMPDIR=\$SNIC_TMP; $PY collect --jobs \$SLURM_CPUS_PER_TASK ${EXTRA:-}" ;;
  *) echo "unknown STEP $STEP" >&2; exit 2 ;;
esac
