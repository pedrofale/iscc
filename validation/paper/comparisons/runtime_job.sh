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
#   STSEED     stCNASim's seed directory (prepared by stcnasim_prep.sh)
# stCNASim simulates spots, not cells: <n_cancer> is converted to spots at OncoSpot's density
# (1,461 spots for 15,000 cells, half malignant), half normal and a quarter per tumour clone.
# <n_cancer> 0 runs stCNASim's own benchmark size instead (600 normal + 2 x 500 tumour spots).
set -uo pipefail
tool=$1; n=$2; seed=$3
echo "TOOL=$tool N_CANCER=$n SEED=$seed NODE=$(hostname) CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
work=$(mktemp -d -p "$OUT" "${tool}_${n}_${seed}_XXXX")
T=(/usr/bin/time -v)
case $tool in
  iscc)
    if   [ "$n" -le 15000 ]; then scale=small; elif [ "$n" -le 60000 ]; then scale=mid; else scale=cm; fi
    # as the paper regime runs it: only the Visium layers materialised, in low-memory mode
    PYTHONPATH=$ISCC_SRC "${T[@]}" "$ISCC_ENV/bin/python" "$COMP/scripts/iscc_runtime.py" "$n" "$seed" "$scale" \
      --layers visium --low-memory ;;
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
  stCNASim)
    if [ "$n" -eq 0 ]; then n0=600; n1=500; n2=500
    else tot=$(( (2 * n * 100 + 513) / 1027 )); n0=$((tot / 2)); n1=$(((tot - n0) / 2)); n2=$((tot - n0 - n1)); fi
    # stCNASim calls samtools through the shell, so its environment's bin must be on PATH, behind
    # stcnasim_shim/samtools, which drops the --threads its BAM merge passes to `samtools cat`
    PATH=$COMP/scripts/stcnasim_shim:$COMP/envs/stcnasim/bin:$PATH "${T[@]}" "$COMP/envs/stcnasim/bin/python" "$COMP/scripts/stcnasim_runtime.py" \
      "$STSEED" $n0 $n1 $n2 "$seed" "$work/st" ;;
  SISTEM)
    m=$(( n < 1000 ? n : 1000 ))
    "${T[@]}" "$COMP/envs/comp/bin/python" "$COMP/scripts/sistem_runtime.py" "$n" "$m" "$seed" "$work/sistem" ;;
esac
rm -rf "$work"
