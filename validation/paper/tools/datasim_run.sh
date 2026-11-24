#!/bin/bash
# Run one other data simulator for the data-simulator supplement (supp_datasim.py) and keep only its
# JSON summary. Meant to run inside a Slurm job from the repository root:
#
#   ISCC_TOOLS_DIR=<images> ISCC_DATASIM_DIR=<summaries> [ISCC_PYTHON=..] [CNASIM=..] \
#     validation/paper/tools/datasim_run.sh <visor|cnasim|splatter|scdesign3|srtsim> [threads]
#
# Raw output (reads, BAMs, count matrices) goes to node-local scratch ($SNIC_TMP, else $TMPDIR) and is
# deleted at the end. Images: visor.sif and datasim-r.sif in $ISCC_TOOLS_DIR (recipes in images/).
# CNASIM is the cnasim executable of an environment with CNAsim 1.3.4 (its python sits beside it).
set -euo pipefail
SIM=$1; N=${2:-${SLURM_CPUS_PER_TASK:-4}}
PY=${ISCC_PYTHON:-python}
: "${ISCC_DATASIM_DIR:?set ISCC_DATASIM_DIR}" "${ISCC_TOOLS_DIR:?set ISCC_TOOLS_DIR}"
export ISCC_DATASIM_DIR PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
SCR=${SNIC_TMP:-${TMPDIR:?set SNIC_TMP or TMPDIR to node-local scratch}}
T=$SCR/datasim-$SIM; rm -rf "$T"; mkdir -p "$T"
trap 'rm -rf "$T"' EXIT
S="$PY validation/paper/supp_datasim.py"
echo "datasim $SIM: commit $(git rev-parse --short HEAD) on $(hostname), $N threads, scratch $T"

case $SIM in
visor)
  # Reads over the span of the 20,000 GIAB HG002 sites (re-derived from the benchmark VCF), at HG002's
  # mean depth, in 4 Mb chunks with 3 kb flanks so a chunk's BAM never outgrows node scratch.
  IMG=$ISCC_TOOLS_DIR/visor.sif
  V() { apptainer exec --cleanenv -B "$SCR" "$IMG" env PATH=/opt/conda/bin:/usr/bin:/bin "$@"; }
  mkdir -p "$T/hap"
  $S hg002-sites "$T/sites.bed" | tee "$T/sites.json"
  FIRST=$($PY -c "import json; print(json.load(open('$T/sites.json'))['first'])")
  LAST=$($PY -c "import json; print(json.load(open('$T/sites.json'))['last'])")
  TARGET=$($PY -c "import numpy as np; print(np.load('validation/data/dna_ref_hg002_bulk_wgs.npz')['coverage'].mean())")
  # GRCh38 chr1 (UCSC hg38, GIAB's naming), upper-cased so soft-masked repeats simulate like the rest.
  curl -s -f -o "$T/chr1.fa.gz" https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz
  gunzip -c "$T/chr1.fa.gz" | awk '/^>/{print; next}{print toupper($0)}' > "$T/chr1.fa"; rm "$T/chr1.fa.gz"
  V samtools faidx "$T/chr1.fa"
  # Diploid haplotypes with no implanted variant: Fig 3b scores copy-number-2 sites only.
  ln "$T/chr1.fa" "$T/hap/h1.fa"; ln "$T/chr1.fa" "$T/hap/h2.fa"
  V samtools faidx "$T/hap/h1.fa"; V samtools faidx "$T/hap/h2.fa"
  # Pilot: SHORtS splits --coverage between the haplotypes; measure, then aim at HG002's mean depth.
  printf "chr1\t2000000\t2200000\t100.0\t100.0\n" > "$T/pilot.bed"
  V VISOR SHORtS -g "$T/chr1.fa" -s "$T/hap" -b "$T/pilot.bed" -o "$T/pilot" --coverage 30 --threads "$N" > /dev/null 2>&1
  PM=$(V samtools depth -a -r chr1:2010000-2190000 "$T/pilot/sim.srt.bam" | awk '{s+=$3;n++}END{print s/n}')
  COV=$(awk -v t="$TARGET" -v m="$PM" 'BEGIN{printf "%.2f", t*30/m}')
  echo "pilot depth at --coverage 30: $PM; HG002 mean $TARGET -> --coverage $COV"
  rm -rf "$T/pilot"; : > "$T/depth.tsv"
  STEP=4000000
  for ((a=FIRST-1; a<LAST; a+=STEP)); do
    b=$((a+STEP))
    awk -v a=$a -v b=$b '$3>a && $3<=b' "$T/sites.bed" > "$T/c.bed"
    [ -s "$T/c.bed" ] || continue
    printf "chr1\t%d\t%d\t100.0\t100.0\n" $((a>3000 ? a-3000 : 0)) $((b+3000)) > "$T/c_region.bed"
    V VISOR SHORtS -g "$T/chr1.fa" -s "$T/hap" -b "$T/c_region.bed" -o "$T/c" --coverage "$COV" \
      --threads "$N" > "$T/c.log" 2>&1 || { tail -20 "$T/c.log"; exit 1; }
    V samtools depth -a -Q 20 -b "$T/c.bed" "$T/c/sim.srt.bam" >> "$T/depth.tsv"
    echo "chunk $a-$b: $(wc -l < "$T/c.bed") sites, scratch $(du -sh "$T" | cut -f1)"
    rm -rf "$T/c"
  done
  $S reduce-wgs "$T/depth.tsv" "$T/sites.bed" VISOR --meta \
    "version=$(apptainer exec "$IMG" cat /opt/versions.txt | tr '\n' ' ')" "image=visor.sif" \
    "genome=GRCh38 chr1 (UCSC hg38), diploid, no implanted variant" "coverage_arg=$COV" \
    "target_mean_depth=$TARGET" "depth=samtools depth -a -Q 20 at the sites" \
    "other=SHORtS defaults (error 0.02, length 150, distance 500, stdev 50, mutation 0.001); 4 Mb chunks, 3 kb flanks" ;;
cnasim)
  # Count mode. Depth and the Lorenz point (CNAsim's coverage non-uniformity) come from DLP+. CNAsim's
  # expected reads per 500 kb bin are cn * C * B * m / (2 R), m the mean of the Beta(a, b) it fits to the
  # Lorenz point (reads.get_alpha_beta); C is set so the mean matches DLP+'s at DLP+'s mean copy number.
  : "${CNASIM:?set CNASIM to the cnasim executable}"
  P=$($S prep-cnasim); echo "DLP+: $P"
  Y=$(echo "$P" | $PY -c "import sys,json; print(round(json.load(sys.stdin)['lorenz_y'], 4))")
  M=$("$(dirname "$CNASIM")/python" -c "from CNAsim.reads import get_alpha_beta; a, b = get_alpha_beta(0.5, $Y); print(a / (a + b))")
  C=$(echo "$P" | $PY -c "import sys,json; d=json.load(sys.stdin); print(round(2*150*d['mean_reads']/(d['mean_cn']*500000*$M), 5))")
  echo "Lorenz y $Y, Beta mean $M -> coverage $C"
  ARGS="-m 1 -n 100 -U -k 50000 -l1 50000 -B 500000 -C $C -X 0.5 -Y $Y -w -v -c 3 -i 10"
  "$CNASIM" $ARGS -o "$T/out" > "$T/cnasim.log"
  $S reduce-cnasim "$T/out" CNAsim --meta "version=CNAsim 1.3.4 (git f034398)" "args=$ARGS" \
    "fit=Lorenz point (0.5, $Y) and mean depth from DLP+ (reads/cn per 500 kb bin, median over cells); W 1 Mb, I 3 at defaults" \
    "karyotype=-w -v -c 3 -i 10: whole-genome doubling and arm/chromosome events over several states, as DLP+ OV2295 (mean cn 4)" ;;
splatter|scdesign3|srtsim)
  IMG=$ISCC_TOOLS_DIR/datasim-r.sif
  RS=(apptainer exec --cleanenv -B "$PWD" -B "$SCR" --env OMP_NUM_THREADS=1 "$IMG" /opt/conda/bin/Rscript)
  mkdir -p "$T/in" "$T/out"
  VER="$(apptainer exec "$IMG" cat /opt/versions.txt | tr '\n' ' ')"
  if [ "$SIM" = srtsim ]; then
    $S prep-visium "$T/in"
    "${RS[@]}" validation/paper/tools/datasim_srtsim.R "$T/in" "$T/out" "$N" 1
    $S reduce-visium "$T/out/srtsim.mtx" "$T/out/srtsim_loc.csv" SRTsim --meta "version=$VER" \
      "image=datasim-r.sif" "input=$(cat "$T/in/prep.json")" "fit=$(cat "$T/out/srtsim.json")"
  else
    $S prep-scrna "$T/in"
    "${RS[@]}" validation/paper/tools/datasim_scrna.R "$T/in" "$T/out" "$SIM" "$N" 1
    NAME=$([ "$SIM" = splatter ] && echo Splatter || echo scDesign3)
    $S reduce-scrna "$T/out/$SIM.mtx" "$NAME" --meta "version=$VER" "image=datasim-r.sif" \
      "input=$(cat "$T/in/prep.json")" "fit=$(cat "$T/out/$SIM.json")"
  fi ;;
*) echo "unknown simulator $SIM" >&2; exit 1 ;;
esac
echo "datasim $SIM done"
