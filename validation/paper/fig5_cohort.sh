#!/bin/bash
# Submit one part of Fig 5 (fig5_cohort.py) as a Slurm job array, one summary file per task.
#
#   PART=part2 BLOCK=10 ISCC_ENV=<conda env> SLURM_ACCOUNT=<account> ISCC_FIG5_OUT=<summary dir> \
#     [ISCC_PAPER_TIER=full ISCC_PAPER_OUT=<registry store>] [ISCC_TOOLS_DIR=<images>] \
#     validation/paper/fig5_cohort.sh
#
# PART is part1 (3 tasks, one cohort each: CPUS patients grown in parallel, then the integrations),
# part2 or part3 (one block of BLOCK patients per task, grown one after the other on one CPU), or
# part2-collect / part3-collect (one job, CPUS fits at a time). Part I needs ISCC_PAPER_TIER=full (the
# sweep scale) and, to reuse R6's tumours for the first cohort, ISCC_PAPER_OUT pointing at the
# registry store that holds R6. N_PATIENTS shrinks the cohorts (smoke tests), EXTRA adds options
# (e.g. "--toy --target 400"). Logs go to $ISCC_FIG5_OUT/logs. Never -N 1.
set -euo pipefail
: "${PART:?set PART}" "${ISCC_ENV:?set ISCC_ENV}" "${SLURM_ACCOUNT:?set SLURM_ACCOUNT}"
: "${ISCC_FIG5_OUT:?set ISCC_FIG5_OUT}"
HERE=$(cd "$(dirname "$0")" && pwd)
export REPO=$(cd "$HERE/../.." && pwd) ISCC_FIG5_OUT PART
LOGS=$ISCC_FIG5_OUT/logs
mkdir -p "$LOGS"
PY="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=\$REPO/src $ISCC_ENV/bin/python \$REPO/validation/paper/fig5_cohort.py"
case "$PART" in
  part1|part2|part3)
    NP=${N_PATIENTS:+--n-patients $N_PATIENTS}
    N=$(PYTHONPATH="$REPO/src" "$ISCC_ENV/bin/python" "$HERE/fig5_cohort.py" count "$PART" --block "${BLOCK:-10}" $NP)
    CPUS=${CPUS:-$([ "$PART" = part1 ] && echo 6 || echo 1)}
    sbatch --parsable -A "$SLURM_ACCOUNT" -p "${PARTITION:-cpu}" -J "fig5-$PART" -n 1 -c "$CPUS" \
      --mem="${MEM:-$([ "$PART" = part1 ] && echo 48G || echo 8G)}" -t "${TIME:-08:00:00}" \
      --array="0-$((N - 1))%${MAX_PARALLEL:-100}" -o "$LOGS/%x-%A_%a.out" --export=ALL \
      --wrap "export TMPDIR=\$SNIC_TMP; $PY $PART \$SLURM_ARRAY_TASK_ID --block ${BLOCK:-10} ${NP:-} ${EXTRA:-}" ;;
  part2-collect|part3-collect)
    sbatch --parsable -A "$SLURM_ACCOUNT" -p "${PARTITION:-cpu}" -J "fig5-$PART" -n 1 -c "${CPUS:-8}" \
      --mem="${MEM:-16G}" -t "${TIME:-12:00:00}" -o "$LOGS/%x-%j.out" --export=ALL \
      --wrap "export TMPDIR=\$SNIC_TMP; $PY $PART --jobs \$SLURM_CPUS_PER_TASK ${EXTRA:-}" ;;
  *) echo "unknown PART $PART" >&2; exit 2 ;;
esac
