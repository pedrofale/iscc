#!/bin/bash
# Submit every tool on every arm of a P2 dataset (make_layer_dataset.py --arms generated,permuted,noise):
# the generated and permuted arms at tool seed 0, the noise arm (the null) at tool seed 1.
#
#   SLURM_ACCOUNT=<account> ISCC_TOOLS_DIR=<dir> validation/paper/tools/submit_p2.sh <dataset> <runs> [tools]
#
# Results go to <runs>/<arm>/, one folder per tool and assay; score them with score_tools.py. Set ARMS to
# submit a subset (e.g. ARMS=noise).
set -euo pipefail
DS=$(readlink -f "$1"); RUNS=$2; TOOLS=${3:-}
HERE=$(dirname "$(readlink -f "$0")")
for arm in ${ARMS:-generated permuted noise}; do
  [ -d "$DS/$arm" ] || continue
  TOOL_SEED=$([ "$arm" = noise ] && echo 1 || echo 0) "$HERE/submit_tools.sh" "$DS/$arm" "$RUNS/$arm" $TOOLS
done
