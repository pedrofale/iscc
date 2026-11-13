#!/bin/bash
# Submit every copy-number tool on one dataset (from make_layer_dataset.py) as separate Slurm jobs.
#
#   SLURM_ACCOUNT=<account> ISCC_TOOLS_DIR=<dir> validation/paper/tools/submit_tools.sh <dataset> <out> [tools]
#
# <dataset> holds visium/ and scrna/; results go to <out>/<tool>-<assay>/ in the common format of
# calls.py, logs to <out>/logs/. [tools] is a comma-separated subset (default: all of them). CalicoST
# runs on Visium only (it needs spots), Numbat on scRNA only; the others run on both assays.
# copykat-ref is CopyKAT given the dataset's normal barcodes as its reference (<out>/copykat-<assay>-ref).
set -euo pipefail
DS=$(readlink -f "$1"); OUT=$2; TOOLS=${3:-copykat,copykat-ref,clonalscope,calicost,cancerfinder,infercnvpy,numbat}
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT}" "${ISCC_TOOLS_DIR:?set ISCC_TOOLS_DIR}"
HERE=$(dirname "$(readlink -f "$0")")
mkdir -p "$OUT/logs"; OUT=$(readlink -f "$OUT")
for tool in ${TOOLS//,/ }; do
  for assay in visium scrna; do
    [ "$tool" = calicost ] && [ "$assay" = scrna ] && continue
    [ "$tool" = numbat ] && [ "$assay" = visium ] && continue
    [ -d "$DS/$assay" ] || continue
    case "$tool" in copykat*|calicost|numbat) c=4; mem=48G ;; *) c=2; mem=32G ;; esac
    if [ "$tool" = copykat-ref ]; then run="copykat $DS/$assay $OUT/copykat-$assay-ref 0 1"; name=copykat-$assay-ref
    else run="$tool $DS/$assay $OUT/$tool-$assay"; name=$tool-$assay; fi
    sbatch --parsable -A "$SLURM_ACCOUNT" -p "${PARTITION:-cpu}" -J "$name" -c "$c" --mem="$mem" \
      -t "${TIME:-04:00:00}" -o "$OUT/logs/$name-%j.out" --export=ALL \
      --wrap "/usr/bin/time -v $HERE/run_tool.sh $run"
  done
done
