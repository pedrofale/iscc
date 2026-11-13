#!/bin/bash
# Run one copy-number tool on a human-gene dataset and write the common output (see calls.py).
#
#   ISCC_TOOLS_DIR=<dir> validation/paper/tools/run_tool.sh <tool> <dataset_dir> <out_dir> [runner args]
#
# <tool> is copykat, clonalscope, calicost, cancerfinder, infercnvpy or numbat (scRNA only); <dataset_dir> is a folder
# written by iscc.data.human.write_visium or write_scrna. ISCC_TOOLS_DIR holds the images built by
# images/build_image.sh (r-cnv.sif, py-tools.sif) and data/ (clonalscope/cyclegenes.rds, and
# cancerfinder/sc_pretrain_article.pkl and st_pretrain_article.pkl, Cancer-Finder's released
# checkpoints). As a Slurm job, call it through --wrap so that it finds its own directory:
#
#   sbatch -A <account> -p cpu -c 4 --mem=32G -t 04:00:00 \
#     --wrap "validation/paper/tools/run_tool.sh copykat ds1/visium out/copykat-visium"
set -euo pipefail
TOOL=$1; DS=$(readlink -f "$2"); OUT=$3; shift 3
HERE=$(dirname "$(readlink -f "$0")")
BIN=$HERE/bin
: "${ISCC_TOOLS_DIR:?set ISCC_TOOLS_DIR to the directory of the tool images}"
export ISCC_TOOLS_DIR
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
CORES=${SLURM_CPUS_PER_TASK:-1}
case "$TOOL" in
  copykat)      "$BIN/rscript-r-cnv" "$HERE/copykat.R" "$DS" "$OUT" "$CORES" "$@" ;;
  clonalscope)  "$BIN/rscript-r-cnv" "$HERE/clonalscope.R" "$DS" "$OUT" "$ISCC_TOOLS_DIR/data/clonalscope" "$@" ;;
  calicost)     "$BIN/python-calicost" "$HERE/calicost.py" "$DS" "$OUT" "$@" ;;
  cancerfinder)
    # Cancer-Finder ships one checkpoint for single cells and one for spatial spots.
    if [ -d "$DS/outs/spatial" ]; then CKPT=st_pretrain_article.pkl; else CKPT=sc_pretrain_article.pkl; fi
    "$BIN/python-cancerfinder" "$HERE/cancerfinder.py" "$DS" "$OUT" "$ISCC_TOOLS_DIR/data/cancerfinder/$CKPT" "$@" ;;
  infercnvpy)   "$BIN/python-infercnv" "$HERE/infercnvpy_layer.py" "$DS" "$OUT" "$@" ;;
  numbat)
    [ -f "$DS/numbat/df_allele.tsv.gz" ] || "$BIN/python-infercnv" "$HERE/numbat_inputs.py" "$DS"
    "$BIN/rscript-r-cnv" "$HERE/numbat_layer.R" "$DS" "$OUT" "$CORES" "$@" ;;
  *) echo "unknown tool $TOOL (copykat, clonalscope, calicost, cancerfinder, infercnvpy, numbat)" >&2; exit 2 ;;
esac
