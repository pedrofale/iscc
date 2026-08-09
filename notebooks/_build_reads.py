"""Builder for notebooks/reads.ipynb — the read-emission demo.

Run with the iscc env to (re)generate the notebook, then execute it with nbconvert. Shows the
synthetic-reference -> per-cell FASTA -> DWGSIM path end-to-end, guarded so it no-ops without
the optional binaries (dwgsim / art / bwa / samtools). Mirrors _build_assay_dna.py's style.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))


md(r"""# DNA read emission: count/coverage matrix -> FASTQ / BAM

Read-level realism for DNA (`iscc.data.reads`), following SISTEM (Weiner & Bansal 2025):
per-cell full reference -> copy-number coverage distribution -> third-party short-read
simulator -> BAM.

```
count/coverage matrix  ──►  Reference{synthetic|real}
   (the universal             │  apply CNAs (dup/del) + SNVs (substitute, allele-aware)
    interface)                ▼
                         per-cell FASTA  ──►  coverage (∝ copy number, breadth-aware)
                                              │  variants.inject(total=coverage, alt=DNA-VAF)
                                              ▼
                              DWGSIM (default) / ART  ──►  FASTQ  ──►  bwa+samtools  ──►  BAM
```

The bespoke layers (reference, per-cell FASTA, coverage, the shared variant seam) run with **no
binaries installed**; only the final shell-out needs `dwgsim`/`art_illumina` (+ `bwa`/`samtools`
for the BAM). This notebook degrades gracefully when they are absent.""")

code(r"""import numpy as np
import pandas as pd

from iscc.data.reads import (
    SyntheticReference, build_cell_fasta, coverage_budget, emit_dna_reads,
    inject, DwgsimAdapter, find_binary,
)

# A tiny ground-truth tumour: one amplified segment (CN8), one deleted (CN0), a few het SNVs.
N_SEG, SZ = 4, 25
GENES = [f"G_{s}_{p}" for s in range(N_SEG) for p in range(SZ)]
n_cells = 30
cnv = np.full((n_cells, len(GENES)), 2.0)
cnv[:, 1 * SZ:2 * SZ] = 8.0     # amplicon on segment 1
cnv[:, 3 * SZ:4 * SZ] = 0.0     # deletion on segment 3
af = np.zeros((n_cells, len(GENES)))
af[:, 5] = 0.5                   # het SNV in diploid seg 0
af[:, 30] = 0.5                  # het SNV in the amplicon
cells = [f"C{i}" for i in range(n_cells)]
cell_data = {
    "cell_cnv": pd.DataFrame(cnv, index=cells, columns=GENES),
    "cell_snv": pd.DataFrame(af, index=cells, columns=GENES),
    "cell_type": pd.DataFrame(["cloneA"] * n_cells, index=cells, columns=["cell_id"]),
}
print("binaries:", {b: find_binary(b) for b in ["dwgsim", "art_illumina", "bwa", "samtools"]})""")

md(r"""## 1. The shared variant seam (`variants.inject`) — total preserved

Generic on `(total, alt_fraction)`: partition the molecules at a locus into alt/ref,
conserving the total exactly. DNA uses `total = coverage (∝CN)` and `alt_fraction = DNA-VAF`;
a later RNA session reuses the same call with `total = UMI count`, `alt_fraction = observed
RNA-VAF`.""")

code(r"""rng = np.random.default_rng(0)
totals = np.array([10, 100, 1000, 5000])
split = inject(totals, alt_fraction=0.3, error_rate=0.001, rng=rng)
print("alt :", split.alt)
print("ref :", split.ref)
print("conserved:", np.array_equal(split.alt + split.ref, totals))
print("observed VAF at depth 5000:", round(split.alt[-1] / totals[-1], 3), "(true 0.30)")""")

md(r"""## 2. Per-cell reference: CNAs duplicate/delete sequence, SNVs substitute bases""")

code(r"""ref = SyntheticReference(GENES, seed=1, locus_length=60)
recs = build_cell_fasta(ref, cnv[0], af[0], "/tmp/C0.fa", name="C0")
from collections import Counter
seg_copies = Counter(k.split("_")[1] for k in recs)
print("copies per segment:", dict(seg_copies))   # seg1 -> 8 (amp), seg3 absent (del), else 2
# het SNV at locus 5 (seg0, CN2): exactly one of the two copies carries the substituted base.
pos = ref.locus_local_pos[5]
print("ref base:", ref.base_seq[0][pos],
      "| copies:", [recs[f"C0_seg0_cp{c}"][pos] for c in range(2)])""")

md(r"""## 3. Coverage budget reuses the C1 model — reads ∝ copy number""")

code(r"""per_seg, assay = coverage_budget(cell_data, breadth="wgs", modality="bulk", seed=2)
print("per-segment reads:", per_seg)
print("amplicon/diploid ratio:", round(per_seg[1] / max(per_seg[0], 1), 2), "(~4x for CN8 vs CN2)")
print("deleted segment reads:", per_seg[3])""")

md(r"""## 4. End-to-end emission (DWGSIM) — guarded

`emit_dna_reads` builds the per-cell FASTA(s), the coverage budget, the allele split and the exact
simulator command; if the binary is installed it shells out to FASTQ (and, with `emit_bam`,
aligns to a sorted/indexed BAM). Without the binary it returns `status="skipped:..."` and the
bespoke artefacts so the rest of the pipeline still runs.""")

code(r"""res = emit_dna_reads(cell_data, simulator="dwgsim", modality="bulk", breadth="wgs",
                 outdir="/tmp/reads_demo", seed=3, emit_bam=True)
print("status            :", res["status"])
print("dwgsim command    :", " ".join(res["command"]))
print("per-cell FASTA     :", res["fasta"])
print("mean coverage      :", res["mean_coverage"])
print("allele split (seg) :", res["allele_split"])
print("fastq              :", res["fastq"] or "(none — binary absent)")
print("bam                :", res["bam"] or "(none — binary absent)")""")

md(r"""**Backends:** synthetic reference (default, implemented) + real-genome reference (seam,
ingests a user FASTA via the same interface). **Simulators:** DWGSIM (default)
+ ART. **To actually emit:** `dwgsim>=0.1.13` (or `art_illumina`) for FASTQ, `bwa` + `samtools`
for the BAM. The `variants.inject` seam is modality-generic and reused unchanged by scRNA.""")

md(r"""## 5. Mutation-aware scRNA reads — and why variant calling from scRNA is hard

The **same** variant seam drives scRNA: `emit_scrna_reads` conserves the assay's UMI totals and, at a
mutated locus, splits UMIs into alt/ref at the observed RNA-VAF (`cell_rna_vaf × obs_fidelity`).
Because a variant is only seen where its gene is **expressed** and captured, and is further
under-detected by the single `obs_fidelity` knob (monoallelic expression / bursting / RT error),
scRNA **misses most true mutations** that scDNA would call — the payoff result for benchmarking
scRNA variant callers. This is the cross-modal consistency in action: the *same* mutation, read out
by DNA vs RNA, gives very different answers, for known reasons.""")

code(r"""from iscc.tumor.models import GenotypeTumor
from iscc.data.reads import emit_scrna_reads

# grow a small tumour so cells carry real, expressed mutations (ground truth)
tum = GenotypeTumor(config="example_config.yaml", seed=2); tum.grow(n_steps=150, seed=2)
cd = tum.cell_data
n_sites = int((cd["cell_snv"].values > 0).sum())
print(f"grew {tum.get_tumor_size()} cells with {n_sites} true (cell, locus) mutation sites")

for f in [0.4, 0.7, 1.0]:
    r = emit_scrna_reads(cd, obs_fidelity=f, protocol="10x", seed=1)
    dna = np.asarray(r["dna_vaf"]); alt = np.asarray(r["alt"]); tot = np.asarray(r["total"])
    site = dna > 0                                   # a true somatic mutation
    detected = site & (tot > 0) & (alt >= 1)         # seen in scRNA (expressed + >=1 alt UMI)
    rate = detected[site].mean()
    print(f"  obs_fidelity={f}: scRNA detects {rate:5.1%} of true mutations "
          f"(scDNA would call them at their genomic VAF)")""")

md(r"""So even at perfect fidelity scRNA recovers only a minority of true mutations — the detection
is **gated by expression**. `validation/validate_scrna_snv.py` renders the full result (observed
RNA-VAF vs true DNA-VAF, detection vs fidelity, detection vs expression). Spot-barcoded **Visium**
reads reuse the same seam (`emit_visium_reads`).""")

nb["cells"] = cells
out = os.path.join(os.path.dirname(__file__), "reads.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out)
