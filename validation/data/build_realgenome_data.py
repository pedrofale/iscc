"""Build the real-genome reference data for M3b (fit-to-real PCAWG + Charm), with provenance.

Downloads four **public, citable** sources and derives the per-arm tables iscc's real-genome
mode (DESIGN_inference A.5) consumes. Mirrors M3a (`build_noble_empirical_indices.py`): download +
document the source, write small derived CSVs under `validation/data/` (un-ignored). Nothing is
fabricated — every number traces to a downloaded source file.

Sources (all hg38 / as published):
  1. **UCSC cytoBand (hg38)** — chromosome-arm lengths and centromere positions.
     https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz
  2. **COSMIC Cancer Gene Census** — gene -> genome location + "Role in Cancer" (oncogene/TSG),
     mapped to chromosome arm to give per-arm oncogene/TSG counts. Sombira/COSMIC; mirrored in the
     CINner repo (Dinh et al. 2025) which is what we download from for reproducibility.
  3. **Davoli et al. 2013 (Cell 155:948)** — per-arm "Charm" score (the orthogonal target) plus
     Davoli's per-arm amplification/deletion frequencies. CINner-mirrored `Davoli_Charm_score.csv`.
  4. **PCAWG consensus copy number** (ICGC/TCGA Pan-Cancer) — per-donor allele-specific CN, from
     which we compute per-arm gain/loss frequencies per cancer type (the fit-to-real CNA target,
     independent of Davoli so the Charm correlation is genuinely orthogonal). CINner-mirrored
     `PCAWG_<type>.csv`.

The arm set is Davoli's canonical 39 autosomal arms (the 5 gene-poor acrocentric p-arms are
excluded), matching CINner.

Outputs (under validation/data/):
  * realgenome_arms.csv    — arm, length_bp, n_onc, n_tsg, charm
  * realgenome_cna.csv     — arm, gain, loss, gain_<TYPE>, loss_<TYPE> for each PCAWG type
  * realgenome_PROVENANCE.md — sources, URLs, citations, derivation notes

Usage:  python validation/data/build_realgenome_data.py
"""
import csv
import gzip
import io
import os
import urllib.request
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CINNER = "https://raw.githubusercontent.com/dinhngockhanh/CINner/master/inst/extdata/"
CYTOBAND = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz"
PCAWG_TYPES = ["BRCA-UK", "OV-AU"]   # breast (default) + ovarian; both CINner-mirrored

AUTOSOMES = [str(i) for i in range(1, 23)]


def _get(url, binary=False, timeout=90):
    data = urllib.request.urlopen(url, timeout=timeout).read()
    return data if binary else data.decode("utf-8", "replace")


# --- 1. cytoBand: arm lengths + centromere positions ------------------------
def cytoband_arms():
    txt = gzip.decompress(_get(CYTOBAND, binary=True)).decode()
    cent, chrom_end = {}, {}
    for ln in txt.splitlines():
        chrom, s, e, name, stain = ln.split("\t")
        if not chrom.startswith("chr"):
            continue
        c = chrom[3:]
        s, e = int(s), int(e)
        chrom_end[c] = max(chrom_end.get(c, 0), e)
        if stain == "acen":
            cs, ce = cent.get(c, (10 ** 12, 0))
            cent[c] = (min(cs, s), max(ce, e))
    return cent, chrom_end


def arm_bounds(arm, cent, chrom_end):
    c = arm[:-1]
    cs, ce = cent[c]
    return (0, cs) if arm.endswith("p") else (ce, chrom_end[c])


def arm_length(arm, cent, chrom_end):
    lo, hi = arm_bounds(arm, cent, chrom_end)
    return hi - lo


# --- 2. COSMIC CGC: per-arm oncogene / TSG counts ---------------------------
def cgc_arm_counts(arms, cent):
    d = _get(CINNER + "cancer_gene_census.csv")
    onc = defaultdict(int)
    tsg = defaultdict(int)
    armset = set(arms)
    for r in csv.DictReader(io.StringIO(d)):
        loc, role = r.get("Genome Location", ""), (r.get("Role in Cancer") or "")
        if not loc or ":" not in loc:
            continue
        chrom = loc.split(":")[0]
        if chrom not in cent:
            continue
        try:
            start = int(loc.split(":")[1].split("-")[0])
        except ValueError:
            continue
        cs, ce = cent[chrom]
        arm = chrom + ("p" if start < cs else "q")
        if arm not in armset:
            continue
        if "oncogene" in role:
            onc[arm] += 1
        if "TSG" in role:
            tsg[arm] += 1
    return onc, tsg


# --- 3. Davoli: arm list + Charm score --------------------------------------
def davoli_charm():
    d = _get(CINNER + "Davoli_Charm_score.csv")
    rows = list(csv.DictReader(io.StringIO(d.lstrip("﻿"))))
    arms = [r["Arm"] for r in rows]
    # Charm(TSG-OG)score is positive for TSG-dominated (deletion-favouring) arms. We store the
    # OG-TSG orientation (negate) so positive charm == amplification-favouring, matching s_arm>1
    # and the (n_onc - n_tsg) content score — an inferred s_arm should *positively* correlate.
    charm = {r["Arm"]: -float(r["Charm(TSG-OG)score"]) for r in rows}
    return arms, charm


# --- 4. PCAWG: per-arm gain/loss frequency per cancer type ------------------
def pcawg_arm_freq(cancer_type, arms, cent, chrom_end):
    """Fraction of donors with an arm-level gain / loss, relative to each donor's ploidy.

    For every donor we compute the genome-wide length-weighted mean total copy number (sample
    ploidy, handling whole-genome doublings), then each arm's length-weighted mean CN; the arm is
    a gain if it exceeds ploidy + 0.5 and a loss if below ploidy - 0.5. Frequencies are over
    donors with coverage of that arm.
    """
    d = _get(CINNER + f"PCAWG_{cancer_type}.csv")
    armset = set(arms)
    donor = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))   # donor -> arm -> [wlen, wlen*cn]
    geno = defaultdict(lambda: [0.0, 0.0])                          # donor -> [wlen, wlen*cn]
    for r in csv.DictReader(io.StringIO(d)):
        c = r["chromosome"]
        if c not in cent:
            continue
        try:
            s, e, cn = int(r["start"]), int(r["end"]), float(r["total_cn"])
        except (ValueError, KeyError):
            continue
        dn = r["donor_unique_id"]
        seglen = max(e - s, 0)
        geno[dn][0] += seglen
        geno[dn][1] += seglen * cn
        cs, ce = cent[c]
        for arm, (lo, hi) in [(c + "p", (0, cs)), (c + "q", (ce, chrom_end[c]))]:
            if arm not in armset:
                continue
            ov = max(min(e, hi) - max(s, lo), 0)
            if ov > 0:
                donor[dn][arm][0] += ov
                donor[dn][arm][1] += ov * cn
    gain = defaultdict(int)
    loss = defaultdict(int)
    cover = defaultdict(int)
    ndonor = 0
    for dn, g in geno.items():
        if g[0] <= 0:
            continue
        ndonor += 1
        ploidy = g[1] / g[0]
        for arm in arms:
            w, wc = donor[dn][arm]
            if w <= 0:
                continue
            cover[arm] += 1
            acn = wc / w
            if acn > ploidy + 0.5:
                gain[arm] += 1
            elif acn < ploidy - 0.5:
                loss[arm] += 1
    gfreq = {a: gain[a] / cover[a] if cover[a] else float("nan") for a in arms}
    lfreq = {a: loss[a] / cover[a] if cover[a] else float("nan") for a in arms}
    return gfreq, lfreq, ndonor


def main():
    print("downloading UCSC cytoBand (hg38)...")
    cent, chrom_end = cytoband_arms()

    print("downloading Davoli Charm scores (arm list + orthogonal target)...")
    arms, charm = davoli_charm()
    print(f"  {len(arms)} arms")

    print("downloading COSMIC Cancer Gene Census (per-arm oncogene/TSG)...")
    onc, tsg = cgc_arm_counts(arms, cent)
    print(f"  mapped {sum(onc.values())} oncogene / {sum(tsg.values())} TSG annotations")

    # realgenome_arms.csv
    arms_path = os.path.join(HERE, "realgenome_arms.csv")
    with open(arms_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "length_bp", "n_onc", "n_tsg", "charm"])
        for a in arms:
            w.writerow([a, arm_length(a, cent, chrom_end), onc.get(a, 0), tsg.get(a, 0),
                        f"{charm.get(a, float('nan')):.6f}"])
    print(f"  -> {arms_path}")

    # realgenome_cna.csv (PCAWG per-type gain/loss; first type is the default columns)
    cols = {}
    ndonors = {}
    for i, ct in enumerate(PCAWG_TYPES):
        print(f"downloading PCAWG {ct} consensus CN (per-arm gain/loss)...")
        gf, lf, nd = pcawg_arm_freq(ct, arms, cent, chrom_end)
        ndonors[ct] = nd
        cols[f"gain_{ct}"] = gf
        cols[f"loss_{ct}"] = lf
        if i == 0:                                  # default (unsuffixed) columns
            cols["gain"], cols["loss"] = gf, lf
        print(f"  {nd} donors")
    cna_path = os.path.join(HERE, "realgenome_cna.csv")
    header = ["arm"] + list(cols.keys())
    with open(cna_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for a in arms:
            w.writerow([a] + [f"{cols[k][a]:.6f}" for k in cols])
    print(f"  -> {cna_path}")

    _write_provenance(arms, ndonors)
    print("done.")


def _write_provenance(arms, ndonors):
    path = os.path.join(HERE, "realgenome_PROVENANCE.md")
    types = ", ".join(f"{ct} (n={ndonors[ct]} donors)" for ct in PCAWG_TYPES)
    with open(path, "w") as f:
        f.write(f"""# Real-genome reference data — provenance (M3b, DESIGN_inference A.5)

Generated by `validation/data/build_realgenome_data.py`. All values are derived from public,
citable sources; nothing is fabricated. Arm set = Davoli's 39 autosomal arms (acrocentric p-arms
excluded).

## Sources
1. **UCSC cytoBand (hg38)** — arm lengths & centromere positions.
   `{CYTOBAND}`
2. **COSMIC Cancer Gene Census** — gene -> "Role in Cancer" (oncogene/TSG), mapped to arm via the
   gene's genome location and the cytoBand centromere -> per-arm oncogene/TSG counts.
   Mirrored at `{CINNER}cancer_gene_census.csv`
   (cite: Sondka et al. 2018, Nat Rev Cancer; COSMIC).
3. **Davoli et al. 2013, Cell 155:948** — per-arm "Charm" score. We store the OG-TSG orientation
   (negated `Charm(TSG-OG)score`) so positive = amplification-favouring, aligned with `s_arm>1`.
   Mirrored at `{CINNER}Davoli_Charm_score.csv`.
4. **PCAWG consensus copy number** (ICGC/TCGA Pan-Cancer Analysis of Whole Genomes) — per-donor
   allele-specific CN. Per-arm gain/loss frequency = fraction of donors whose arm mean CN is
   above (gain) / below (loss) the donor's ploidy by >0.5. Types: {types}.
   Mirrored at `{CINNER}PCAWG_<type>.csv` (cite: ICGC/TCGA PCAWG, Nature 2020).

## Outputs
- `realgenome_arms.csv` — `arm, length_bp, n_onc, n_tsg, charm`
- `realgenome_cna.csv` — `arm, gain, loss` (default = {PCAWG_TYPES[0]}) plus `gain_<TYPE>/loss_<TYPE>`

## Use
`iscc.inference.genome.load_default()` builds a `GenomeSpec` from `realgenome_arms.csv`;
`load_real_cna_profile(cancer_type=...)` loads the fit-to-real target from `realgenome_cna.csv`.
The Charm correlation (A.4.3) is genuinely orthogonal: `s_arm` is fit to PCAWG CN, then correlated
against the independent Davoli Charm score.
""")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
