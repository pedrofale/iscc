"""Build the real DNA reference data for the M4 DNA `estimate_dna()` validation, with provenance.

The DNA analogue of how scRNA validation fits PBMC3k and Visium fits scanpy's `visium_sge`: fetch
a **real, public, citable** DNA-seq dataset per regime, REDUCE it to the small reusable inputs
`estimate_dna` consumes — per-locus / per-cell **coverage** + **alt** counts + a per-locus **called
copy number** (+ het / variant masks where available) — and cache it under `validation/data/` as a
compact `.npz`. Raw sources are 100s of MB–GB; we never require/keep them — only the small reduction
is cached. `validate_dna.py` loads the cache and falls back to the synthetic round-trip when a
dataset is absent (offline / CI), exactly like `validate_visium.py`'s `--synthetic`.

One dataset per DNA regime (chosen as the field's gold standards):

  BULK WGS — **GIAB HG002** (NIST Genome-in-a-Bottle, AshkenazimTrio son NA24385).
    The WGS technical gold standard. We stream the small-variant **benchmark VCF** (v4.2.1, GRCh38)
    and read its per-site `AD` (net allele depths) + `GT`: coverage = ref+alt depth, alt = alt
    depth, CN = 2 everywhere (a normal diploid genome -> trivial CN-conditioning), het = GT 0/1.
    Fits: mu_depth, kappa / nb_dispersion (CN=2 coverage dispersion), het BAF. error_rate / GC are
    prior-only (no hom-ref sites / no reference FASTA in the reduction) — flagged honestly.
    Source: https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/
            HG002_NA24385_son/NISTv4.2.1/GRCh38/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
    Cite: Zook et al. 2019, Nat Biotechnol (GIAB); HG002 NIST v4.2.1 benchmark.

  SC PANEL — **Mission Bio Tapestri** (targeted scDNA; deep per-amplicon, clear het calls).
    The richest single-cell regime. We read a public Tapestri `.h5` (KaryoTap, Mays et al.) — its
    `dna_variants` layers carry per-cell x per-variant `DP` (depth), `AF` (alt %, 0–100), and `NGT`
    (genotype 0/1/2/3); `dna_read_counts` carries per-amplicon reads for CN. Reduce to
    coverage = DP, alt = round(AF/100 * DP), CN = amplicon-depth-normalized (2 x reads / median),
    het = NGT==1. Fits the full single-cell layer: mu_depth (deep), kappa, capture_sigma (per
    amplicon), **ado_rate + beta_binom_conc** (the het-collapse layer only single-cell data informs).
    Source: Zenodo 11094529 `tapestri-experiment01-panelv1.h5` (~406 MB; not auto-downloaded).
    Cite: Mays et al. 2024, "KaryoTap Enables Aneuploidy Detection in Thousands of Single Cells",
          Zenodo 10.5281/zenodo.11094529.

  SC WGS — **DLP+ OV2295** (Shah lab; the scWGS copy-number benchmark, Laks 2019).
    Genome-wide, very low depth. We stream the per-cell x 500 kb-bin table and read `reads`
    (coverage) + HMMcopy `state` (called CN). Reduce to coverage[cell, bin] + cn[cell, bin] (no
    per-bin alleles -> no alt). Fits: mu_depth (low, in bin units), kappa / nb_dispersion
    (CN-conditioned across the many CN states DLP carries). ado / beta_binom / error are prior-only
    (no het read counts) — flagged honestly.
    Source: Zenodo 3445364 `ov2295_cell_cn.csv.gz` (~172 MB; streamed, early-stopped).
    Cite: Laks et al. 2019, Cell 179:1207 (DLP+); the `signals`/`scgenome` ecosystem.

Robustness: every fetch is wrapped — on any failure (offline, host down, h5 absent) it returns
``None`` with a clear note and the caller falls back. The pure REDUCERS (``reduce_*``) are
network-free and unit-tested on tiny fixtures (`tests/test_dna_reference.py`).

Usage:
    python validation/data/build_dna_reference.py                 # HG002 + DLP (streamable)
    python validation/data/build_dna_reference.py --download-tapestri   # + Tapestri (406 MB)
    python validation/data/build_dna_reference.py --tapestri-h5 PATH.h5 # reduce a local Tapestri h5
"""
import argparse
import os
import urllib.request
import zlib
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache")          # transient raw downloads (git-ignored), never required

GIAB_VCF = ("https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/"
            "HG002_NA24385_son/NISTv4.2.1/GRCh38/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz")
DLP_CSV = "https://zenodo.org/records/3445364/files/ov2295_cell_cn.csv.gz?download=1"
TAPESTRI_URL = ("https://zenodo.org/records/11094529/files/"
                "tapestri-experiment01-panelv1.h5?download=1")
TAPESTRI_CACHE = os.path.join(CACHE, "tapestri-experiment01-panelv1.h5")

# Reference registry: name -> (filename, modality, breadth). validate_dna loads these.
REFERENCES = {
    "hg002":    ("dna_ref_hg002_bulk_wgs.npz", "bulk", "wgs"),
    "tapestri": ("dna_ref_tapestri_sc_panel.npz", "sc", "panel"),
    "dlp":      ("dna_ref_dlp_sc_wgs.npz", "sc", "wgs"),
}


# --------------------------------------------------------------------------------------
# Streaming gunzip (handles both standard gzip and BGZF's concatenated members), line-yielding
# with early stop so we never pull a whole multi-hundred-MB file.
# --------------------------------------------------------------------------------------
def _stream_gzip_lines(url, max_bytes=None, timeout=90):
    """Yield decoded text lines from a remote .gz URL, decompressing on the fly.

    Re-inits the decompressor at each gzip member boundary (BGZF concatenates many members), so it
    works for the BGZF benchmark VCF and the plain-gzip DLP CSV alike. ``max_bytes`` caps the
    *compressed* bytes read so a huge source is never fully downloaded.
    """
    resp = urllib.request.urlopen(url, timeout=timeout)
    dobj = zlib.decompressobj(31)
    buf = ""
    read = 0
    try:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            read += len(chunk)
            data = chunk
            while data:
                out = dobj.decompress(data)
                buf += out.decode("utf-8", "replace")
                if dobj.unused_data:                 # next gzip member (BGZF)
                    data = dobj.unused_data
                    dobj = zlib.decompressobj(31)
                else:
                    data = b""
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                yield line
            if max_bytes is not None and read > max_bytes:
                break
    finally:
        resp.close()


# --------------------------------------------------------------------------------------
# Pure reducers (network-free; unit-tested)
# --------------------------------------------------------------------------------------
def parse_giab_record(fields):
    """Parse one GIAB benchmark VCF data line (tab-split) -> (cov, alt, is_het) or None.

    Keeps only PASS biallelic SNVs with an ``AD`` (ref,alt net depth) and ``GT``; coverage is the
    summed allele depth (consistent with the alt count for a clean BAF), CN is 2 (diploid).
    """
    if len(fields) < 10:
        return None
    _chrom, _pos, _id, ref, alt, _qual, filt, _info, fmt, smp = fields[:10]
    if filt not in ("PASS", "."):
        return None
    if len(ref) != 1 or len(alt) != 1:               # biallelic SNV only (clean het BAF)
        return None
    d = dict(zip(fmt.split(":"), smp.split(":")))
    if "AD" not in d or "GT" not in d:
        return None
    try:
        ad = [int(x) for x in d["AD"].split(",")]
    except ValueError:
        return None
    if len(ad) < 2:
        return None
    cov = ad[0] + ad[1]
    if cov <= 0:
        return None
    gt = d["GT"].replace("|", "/")
    return cov, ad[1], gt in ("0/1", "1/0")


def reduce_dlp_records(cell_rows):
    """Reduce DLP+ per-cell bin rows -> (coverage, cn) matrices (cells x bins).

    ``cell_rows`` is an ordered ``{cell_id: [((chrom, start), reads, state), ...]}``. Bins are the
    fixed 500 kb genome binning shared across cells; we key on (chrom, start) and use the first
    cell's bin set as the column order, dropping bins absent / all-zero across cells.
    """
    if not cell_rows:
        return np.zeros((0, 0)), np.zeros((0, 0))
    ref = next(iter(cell_rows.values()))
    binkeys = [b[0] for b in ref]
    binidx = {k: i for i, k in enumerate(binkeys)}
    n_cells, n_bins = len(cell_rows), len(binkeys)
    cov = np.zeros((n_cells, n_bins), dtype=float)
    cn = np.full((n_cells, n_bins), 2.0, dtype=float)
    for ci, rows in enumerate(cell_rows.values()):
        for key, reads, state in rows:
            j = binidx.get(key)
            if j is not None:
                cov[ci, j] = reads
                cn[ci, j] = state
    keep = (cov.sum(axis=0) > 0) & np.isfinite(cn).all(axis=0)
    return cov[:, keep], cn[:, keep]


def reduce_tapestri(dp, af_pct, ngt, var_amplicon, amplicon_reads, amplicon_ids,
                    cn_clip=8.0):
    """Reduce Tapestri DNA-variant + amplicon-read arrays -> (coverage, alt, cn, het_mask).

    coverage = DP; alt = round(AF/100 * DP) (AF is a 0–100 percentage); CN per (cell, variant) from
    the variant's amplicon: 2 x (per-cell-normalized amplicon depth) / its cross-cell median, clipped
    to [0, ``cn_clip``]. Missing genotypes (NGT==3) get zero coverage so they drop out of every fit.

    het_mask marks GERMLINE-het LOCI broadcast over all covered cells — NOT the per-cell `NGT==1`
    genotype calls. ADO is exactly the cells where a constitutive het collapses to monoallelic, and
    those cells are genotyped HOMOZYGOUS (NGT!=1), so masking on `NGT==1` selects only the cells that
    did NOT drop out and the ADO estimate collapses to ~0. Instead, a variant is called germline-het
    when it is genotyped het in a substantial fraction of covered cells AND its pseudobulk BAF is
    balanced (excludes true-homozygous / strongly-skewed somatic sites whose "collapse" is not ADO).
    `estimate_dna._fit_ado_and_conc` then measures the per-cell monoallelic fraction at those loci
    (de-biased for the Beta-Binomial floor) -> a real Tapestri-scale ADO.
    """
    dp = np.asarray(dp, dtype=float).copy()
    alt = np.rint(np.asarray(af_pct, dtype=float) / 100.0 * dp).astype(int)
    ngt = np.asarray(ngt, dtype=int)
    amp_index = {a: i for i, a in enumerate(list(amplicon_ids))}
    col = np.array([amp_index.get(a, -1) for a in var_amplicon])
    reads = np.asarray(amplicon_reads, dtype=float)
    cell_tot = reads.sum(axis=1, keepdims=True)
    cell_tot[cell_tot == 0] = 1.0
    norm = reads / cell_tot
    med = np.median(norm, axis=0, keepdims=True)
    med[med == 0] = np.nan
    cn_ampl = 2.0 * norm / med                        # cells x amplicons
    cn = np.clip(np.nan_to_num(cn_ampl[:, col], nan=2.0), 0.0, cn_clip)
    cn[:, col < 0] = 2.0
    missing = ngt == 3
    dp[missing] = 0.0
    alt[missing] = 0
    het_mask = germline_het_mask(dp.astype(int), alt, ngt == 1)
    return dp.astype(int), alt, cn, het_mask


def germline_het_mask(coverage, alt, per_cell_het, min_het_frac=0.30, baf_lo=0.30, baf_hi=0.70):
    """Boolean (cells x variants) mask of GERMLINE-het loci over covered cells (for ADO fitting).

    A variant is germline-het if it is genotyped het (``per_cell_het``) in >= ``min_het_frac`` of its
    covered cells AND its pseudobulk BAF (summed alt / summed coverage) is balanced
    (``[baf_lo, baf_hi]``). The mask is that per-locus call broadcast over every covered cell, so the
    ADO estimator sees the DROPOUT cells (collapsed to ~0/1) — not just the successfully-genotyped
    hets. Stateless; also used to migrate an already-reduced cache without re-downloading the raw h5.
    """
    coverage = np.asarray(coverage); alt = np.asarray(alt)
    covered = coverage > 0
    het_frac = per_cell_het.sum(0) / np.maximum(covered.sum(0), 1)
    pb_baf = alt.sum(0) / np.maximum(coverage.sum(0), 1)
    germline = (het_frac >= min_het_frac) & (pb_baf >= baf_lo) & (pb_baf <= baf_hi)
    return covered & germline[None, :]


# --------------------------------------------------------------------------------------
# Fetchers (network; graceful — return None on any failure)
# --------------------------------------------------------------------------------------
def fetch_hg002_bulk(max_sites=20000, max_bytes=80 << 20):
    """Stream the GIAB HG002 benchmark VCF and reduce to bulk-WGS coverage/alt/cn/het."""
    cov, alt, het = [], [], []
    try:
        for line in _stream_gzip_lines(GIAB_VCF, max_bytes=max_bytes):
            if not line or line.startswith("#"):
                continue
            rec = parse_giab_record(line.split("\t"))
            if rec is None:
                continue
            c, a, is_het = rec
            cov.append(c); alt.append(a); het.append(is_het)
            if len(cov) >= max_sites:
                break
    except Exception as e:
        print(f"[HG002 fetch failed: {e}]")
        return None
    if len(cov) < 100:
        print(f"[HG002: only {len(cov)} sites parsed — skipping]")
        return None
    cov = np.asarray(cov, dtype=np.int32)
    return dict(
        modality="bulk", breadth="wgs", depth_model="dm",
        coverage=cov, alt=np.asarray(alt, dtype=np.int32),
        cn=np.full(cov.shape, 2.0, dtype=np.float32),     # diploid normal genome
        het_mask=np.asarray(het, dtype=bool),
        variant_mask=np.ones(cov.shape, dtype=bool),      # every benchmark site is a variant
        source="GIAB HG002 NIST v4.2.1 benchmark VCF (GRCh38); Zook et al. 2019 Nat Biotechnol",
    )


def fetch_dlp_sc(max_cells=40, max_bytes=120 << 20):
    """Stream DLP+ OV2295 per-cell CN bins and reduce to sc-WGS coverage/cn."""
    cell_rows = OrderedDict()
    header_seen = False
    try:
        for line in _stream_gzip_lines(DLP_CSV, max_bytes=max_bytes):
            if not header_seen:
                header_seen = True            # cell_id,sample_id,library_id,chr,start,end,reads,copy,state
                continue
            f = line.split(",")
            if len(f) < 9:
                continue
            cell = f[0]
            if cell not in cell_rows:
                if len(cell_rows) >= max_cells:
                    break
                cell_rows[cell] = []
            try:
                cell_rows[cell].append(((f[3], int(f[4])), int(f[6]), int(f[8])))
            except ValueError:
                continue
    except Exception as e:
        print(f"[DLP fetch failed: {e}]")
        return None
    cov, cn = reduce_dlp_records(cell_rows)
    if cov.shape[0] < 2 or cov.shape[1] < 50:
        print(f"[DLP: reduced to {cov.shape} — skipping]")
        return None
    return dict(
        modality="sc", breadth="wgs", depth_model="dm",
        coverage=cov.astype(np.int32), alt=np.zeros(cov.shape, dtype=np.int32),
        cn=cn.astype(np.float32), het_mask=None,
        variant_mask=None,                                # no per-bin alleles in low-pass scWGS
        source="DLP+ OV2295 (Shah lab; Laks et al. 2019 Cell); Zenodo 3445364 ov2295_cell_cn",
    )


def fetch_tapestri_sc(h5_path=None, max_cells=800, download=False):
    """Reduce a Tapestri `.h5` (local, cached, or downloaded) to sc-panel coverage/alt/cn/het."""
    try:
        import h5py
    except Exception as e:
        print(f"[Tapestri: h5py unavailable: {e}]")
        return None
    path = h5_path or TAPESTRI_CACHE
    if not os.path.exists(path):
        if not download:
            print(f"[Tapestri h5 not found at {path}; pass --tapestri-h5 or --download-tapestri]")
            return None
        os.makedirs(CACHE, exist_ok=True)
        print(f"downloading Tapestri h5 (~406 MB) -> {path} ...")
        try:
            urllib.request.urlretrieve(TAPESTRI_URL, path)
        except Exception as e:
            print(f"[Tapestri download failed: {e}]")
            return None
    try:
        with h5py.File(path, "r") as f:
            dv = f["assays"]["dna_variants"]
            rc = f["assays"]["dna_read_counts"]
            keep = np.where(dv["ca"]["filtered"][:] == 0)[0]     # curated, analysis-ready variants
            nc = min(max_cells, dv["layers"]["DP"].shape[0])
            dp = dv["layers"]["DP"][:nc, :][:, keep]
            af = dv["layers"]["AF"][:nc, :][:, keep]
            ngt = dv["layers"]["NGT"][:nc, :][:, keep]
            var_amplicon = dv["ca"]["amplicon"][:][keep]
            reads = rc["layers"]["read_counts"][:nc, :]
            amplicon_ids = rc["ca"]["id"][:]
    except Exception as e:
        print(f"[Tapestri h5 read failed: {e}]")
        return None
    cov, alt, cn, het = reduce_tapestri(dp, af, ngt, var_amplicon, reads, amplicon_ids)
    return dict(
        modality="sc", breadth="panel", depth_model="dm",
        coverage=cov.astype(np.int32), alt=alt.astype(np.int32), cn=cn.astype(np.float32),
        het_mask=het, variant_mask=None,
        source="Mission Bio Tapestri (KaryoTap exp01 panelv1); Mays et al. 2024, Zenodo 11094529",
    )


# --------------------------------------------------------------------------------------
# Cache I/O
# --------------------------------------------------------------------------------------
def save_reference(path, ref):
    """Save a reduced reference dict to a compact .npz (arrays + string/scalar metadata)."""
    arrs = {}
    for k, v in ref.items():
        if v is None:
            continue
        arrs[k] = np.asarray(v)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrs)


def load_reference(path):
    """Load a reduced reference .npz -> dict with python-typed metadata (or None if absent)."""
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    out = {}
    for k in z.files:
        v = z[k]
        if v.ndim == 0:
            out[k] = v.item().decode() if isinstance(v.item(), bytes) else v.item()
        else:
            out[k] = v
    z.close()
    out.setdefault("het_mask", None)
    out.setdefault("variant_mask", None)
    return out


def reference_path(name):
    return os.path.join(HERE, REFERENCES[name][0])


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=["hg002", "dlp"],
                    choices=list(REFERENCES) + ["all"],
                    help="Which references to build (default: the two streamable ones).")
    ap.add_argument("--max-sites", type=int, default=20000, help="HG002: cap on parsed VCF sites.")
    ap.add_argument("--max-cells", type=int, default=800, help="Tapestri/DLP: cap on cells.")
    ap.add_argument("--tapestri-h5", default=None, help="Path to a local Tapestri .h5 to reduce.")
    ap.add_argument("--download-tapestri", action="store_true",
                    help="Download the 406 MB Tapestri h5 to the cache if not present.")
    args = ap.parse_args()
    names = list(REFERENCES) if "all" in args.datasets else args.datasets

    built = {}
    for name in names:
        print(f"\n=== building {name} ({REFERENCES[name][1]}-{REFERENCES[name][2]}) ===")
        if name == "hg002":
            ref = fetch_hg002_bulk(max_sites=args.max_sites)
        elif name == "dlp":
            ref = fetch_dlp_sc(max_cells=min(args.max_cells, 40))
        elif name == "tapestri":
            ref = fetch_tapestri_sc(h5_path=args.tapestri_h5, max_cells=args.max_cells,
                                    download=args.download_tapestri)
        else:
            ref = None
        if ref is None:
            print(f"  -> {name} unavailable; skipped (validate_dna will fall back).")
            continue
        path = reference_path(name)
        save_reference(path, ref)
        cov = ref["coverage"]
        built[name] = ref["source"]
        sz = os.path.getsize(path) / 1e6
        print(f"  coverage shape {cov.shape}  mean {float(cov[cov > 0].mean()):.1f}  -> {path} "
              f"({sz:.2f} MB)")

    _write_provenance(built)
    print("\ndone.")


def _write_provenance(built):
    path = os.path.join(HERE, "dna_reference_PROVENANCE.md")
    lines = [
        "# Real DNA reference data — provenance (M4 DNA, DESIGN_inference §C.1)\n",
        "Generated by `validation/data/build_dna_reference.py`. Each dataset is a public, citable "
        "DNA-seq gold standard, REDUCED to the small (coverage, alt, called-CN, het) inputs "
        "`estimate_dna` consumes and cached as a compact `.npz`. Raw sources (100s of MB–GB) are "
        "never committed or required; `validate_dna.py` loads the cache and falls back to the "
        "synthetic round-trip when absent.\n",
        "## Datasets (one per DNA regime)\n",
        "1. **BULK WGS — GIAB HG002** (NIST GiaB, NA24385 son). Small-variant benchmark VCF "
        f"(v4.2.1, GRCh38); per-site `AD`+`GT` -> coverage/alt, CN=2, het=GT(0/1).\n   `{GIAB_VCF}`\n"
        "   Fits mu_depth, kappa/nb_dispersion, het BAF; error_rate/GC prior-only (no hom-ref / "
        "reference FASTA in the reduction). Cite: Zook et al. 2019, Nat Biotechnol.\n",
        "2. **SC PANEL — Mission Bio Tapestri** (KaryoTap exp01 panelv1). `dna_variants` DP/AF/NGT "
        "+ `dna_read_counts` per-amplicon reads -> coverage/alt/het + amplicon-normalized CN.\n"
        f"   `{TAPESTRI_URL}` (~406 MB; not auto-downloaded).\n"
        "   Fits the full single-cell layer incl. **ado_rate + beta_binom_conc** + capture_sigma. "
        "Cite: Mays et al. 2024, Zenodo 10.5281/zenodo.11094529.\n",
        "3. **SC WGS — DLP+ OV2295** (Shah lab). Per-cell 500 kb-bin `reads` + HMMcopy `state` -> "
        "coverage/cn (no per-bin alleles).\n"
        f"   `{DLP_CSV}` (~172 MB; streamed, early-stopped).\n"
        "   Fits mu_depth (low, bin units), kappa/nb_dispersion across the many CN states; "
        "ado/beta_binom/error prior-only. Cite: Laks et al. 2019, Cell 179:1207.\n",
        "## Outputs (validation/data/)\n",
    ]
    for name, (fn, mod, br) in REFERENCES.items():
        tag = "built" if name in built else "not built here"
        lines.append(f"- `{fn}` — {mod}-{br} ({tag})")
    lines.append("\n## Use\n`validate_dna.py` loads these via "
                 "`build_dna_reference.load_reference()`; `estimate_dna` is fit on the reduction, "
                 "the technical layer re-simulated, and the same summaries overlaid real vs re-sim.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
