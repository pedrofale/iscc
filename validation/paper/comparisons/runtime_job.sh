#!/bin/bash
# One timed run for the runtime comparison: runtime_job.sh <tool> <n_cancer> <seed>
#
# Each tool runs whole under /usr/bin/time -v on one core; the log's header records the tool, the
# size, the seed, the node and its CPU, and runtime_collect.py reads the rest. Paths come from the
# environment (set by the submitting script):
#   COMP       competitors root (OncoSpot, CINner, demon, envs/comp, envs/rcin, scripts)
#   ISCC_ENV   the iscc conda environment
#   ISCC_SRC   iscc's src/ with the engine options (PR #12)
#   ISCC_PAPER_DIR  an iscc checkout whose validation/paper holds the paper regime
#   OUT        where tools may write their outputs (removed after each run)
set -uo pipefail
tool=$1; n=$2; seed=$3
echo "TOOL=$tool N_CANCER=$n SEED=$seed NODE=$(hostname) CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
work=$(mktemp -d -p "$OUT" "${tool}_${n}_${seed}_XXXX")
T=(/usr/bin/time -v)
case $tool in
  iscc)
    if   [ "$n" -le 15000 ]; then scale=small; elif [ "$n" -le 60000 ]; then scale=mid; else scale=cm; fi
    PYTHONPATH=$ISCC_SRC "${T[@]}" "$ISCC_ENV/bin/python" "$COMP/scripts/iscc_runtime.py" "$n" "$seed" "$scale" ;;
  OncoSpot)
    # OncoSpot's malignant share is its tumour fraction (0.5) of all cells.
    cells=$((2 * n))
    cfg="$work/config.yaml"
    "$COMP/envs/comp/bin/python" - "$COMP/OncoSpot/configs/examples/hgsoc_validation_15k_hallmark_v1.yaml" "$cfg" "$cells" "$seed" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c["simulation"]["n_cells"] = int(sys.argv[3]); c["simulation"]["seed"] = int(sys.argv[4])
yaml.safe_dump(c, open(sys.argv[2], "w"))
PY
    run="$COMP/OncoSpot/results/runs/rt_${n}_${seed}_$(date +%s%N)"
    export PATH=$COMP/envs/comp/bin:$PATH ONCOSPOT_PYTHON=$COMP/envs/comp/bin/python \
           ONCOSPOT_CNASIM=$COMP/envs/comp/bin/cnasim ONCOSPOT_DEMON_BIN=$COMP/demon_model/build/demon
    (cd "$COMP/OncoSpot" && "${T[@]}" bash scripts/pipeline/run_cancer_simulation.sh "$cfg" "$run")
    status=$?
    if [ $status -ne 0 ]; then
      echo "FAILED"; tail -20 "$run"/cna/cnasim.log 2>/dev/null   # keep the cause before cleaning up
    fi
    rm -rf "$run" ;;
  CINner)
    m=$(( n < 1000 ? n : 1000 ))
    "${T[@]}" "$COMP/envs/rcin/bin/Rscript" "$COMP/scripts/cinner_runtime.R" "$COMP/CINner" "$n" "$m" "$seed" "$work/out.json"
    cat "$work/out.json" 2>/dev/null; echo ;;
  SISTEM)
    m=$(( n < 1000 ? n : 1000 ))
    "${T[@]}" "$COMP/envs/comp/bin/python" "$COMP/scripts/sistem_runtime.py" "$n" "$m" "$seed" "$work/sistem" ;;
esac
rm -rf "$work"
