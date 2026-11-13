#!/bin/bash
# Submit every tool on both arms of a P2 dataset (make_layer_dataset.py --arms generated,permuted).
#
#   SLURM_ACCOUNT=<account> ISCC_TOOLS_DIR=<dir> validation/paper/tools/submit_p2.sh <dataset> <runs> [tools]
#
# Results go to <runs>/generated/ and <runs>/permuted/, one folder per tool and assay; score them with
# score_tools.py (score on each arm, compare between them).
set -euo pipefail
DS=$(readlink -f "$1"); RUNS=$2; TOOLS=${3:-}
HERE=$(dirname "$(readlink -f "$0")")
for arm in generated permuted; do
  "$HERE/submit_tools.sh" "$DS/$arm" "$RUNS/$arm" $TOOLS
done
