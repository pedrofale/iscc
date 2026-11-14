#!/bin/bash
# Build one tool image as a single Apptainer file, without root or fakeroot.
#
#   sbatch -A <account> -p <partition> -c 8 --mem=32G -t 04:00:00 build_image.sh <name> <out_dir>
#
# Pulls a micromamba base image into a writable sandbox on node-local scratch, runs the recipe
# `<name>.sh` from this directory inside it (an unprivileged user owns every file of a sandbox it
# extracted, so the recipe can install into it), then packs the sandbox into <out_dir>/<name>.sif.
# Node-local scratch keeps the sandbox's many files off shared storage; the .sif is one file.
set -euo pipefail
NAME=$1; OUT=$2
# Under sbatch the script runs from a spool copy, so the recipes are found in RECIPES, else the
# submission directory, else beside this script.
HERE=${RECIPES:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
[ -f "$HERE/$NAME.sh" ] || { echo "no recipe $HERE/$NAME.sh" >&2; exit 1; }
BASE=${BASE_IMAGE:-docker://mambaorg/micromamba:1.5.8}
LOCAL=${SNIC_TMP:-${TMPDIR:-/tmp}}/$USER-build-$NAME
rm -rf "$LOCAL"; mkdir -p "$LOCAL/cache" "$LOCAL/tmp"
export APPTAINER_CACHEDIR=$LOCAL/cache APPTAINER_TMPDIR=$LOCAL/tmp
apptainer build --sandbox "$LOCAL/sb" "$BASE"
mkdir -p "$LOCAL/sb/recipes"                 # a writable sandbox cannot create a bind's mount point
apptainer exec --writable --cleanenv --no-home -B "$HERE:/recipes" --env MAMBA_ROOT_PREFIX=/opt/conda \
  "$LOCAL/sb" bash -c "export HOME=/tmp; bash -euo pipefail /recipes/$NAME.sh"
mkdir -p "$OUT"
apptainer build --force "$OUT/$NAME.sif" "$LOCAL/sb"
apptainer exec "$OUT/$NAME.sif" bash -c 'cat /opt/versions.txt' > "$OUT/$NAME.versions.txt"
rm -rf "$LOCAL"
echo "built $OUT/$NAME.sif"
