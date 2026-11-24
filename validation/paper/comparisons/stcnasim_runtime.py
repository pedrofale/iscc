"""stCNASim end to end, for the runtime comparison.

    python stcnasim_runtime.py <seed dir> <n_normal> <n_tumour1> <n_tumour2> <seed> <out dir> [<chroms>]

Runs stCNASim's ``main_wrapper`` (pp, afc, cs, rs) on one core, from the seed prepared by
``stcnasim_prep.sh``: 600 seed spots, two tumour clones with its own ``1n2t_aleloh`` benchmark
copy-number profile, the simulated spots split as given. stCNASim simulates spots, not cells; each
simulated spot is one clone. Prints a JSON line with the stage times and the output size; the
output BAM is deleted afterwards. ``xf_tag`` is off because STARsolo does not write Cell Ranger's
``xf`` tag (stCNASim drops reads without it otherwise); UMIs are Visium v1's 12 bases. ``chroms``
(default: the autosomes) restricts the run, for a quick check of the whole pipeline on one chromosome.
"""
import json
import os
import shutil
import sys
import time


def main():
    seed_dir, n0, n1, n2, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), \
        int(sys.argv[5]), sys.argv[6]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stcnasim_inputs as SI
    from stcnasim import main_wrapper

    os.makedirs(out, exist_ok=True)
    clone_fn = os.path.join(out, "clone_anno.tsv")
    SI.clones(n0, n1, n2, clone_fn)
    t0 = time.time()
    main_wrapper(
        sam_fn=os.path.join(seed_dir, "solo", "Aligned.sortedByCoord.out.bam"),
        cell_anno_fn=os.path.join(seed_dir, "seed_cell_anno.tsv"),
        feature_fn=os.path.join(seed_dir, "features.tsv"),
        phased_snp_fn=os.path.join(seed_dir, "phased_snps.vcf"),
        clone_anno_fn=clone_fn,
        cna_profile_fn=os.path.join(seed_dir, "cna_profile.tsv"),
        refseq_fn=os.path.join(seed_dir, "genome.fa"),
        out_dir=out, umi_len=12, strandness="forward", xf_tag=None, ncores=1, seed=seed,
        chroms=sys.argv[7] if len(sys.argv) > 7 else "human_autosome",
    )
    wall = time.time() - t0
    stages = {}
    for k in ("1_pp", "2_afc", "3_cs", "4_rs"):
        p = os.path.join(out, k)
        if os.path.isdir(p):
            stages[k] = max(os.path.getmtime(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
    bam_dir = os.path.join(out, "4_rs", "bam")
    bam_bytes = sum(os.path.getsize(os.path.join(bam_dir, f)) for f in os.listdir(bam_dir)) \
        if os.path.isdir(bam_dir) else 0
    rec = dict(tool="stCNASim", spots=n0 + n1 + n2, n_normal=n0, n_tumour=n1 + n2, seed=seed,
               total_s=wall, bam_mb=bam_bytes / 1e6,
               stage_end_s={k: v - t0 for k, v in stages.items()})
    print(json.dumps(rec))
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
