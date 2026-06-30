"""Cross-modality allele-consistency tests (the shared canonical reference, F7/F7b).

The engine genome is base-agnostic (a "position L mutated?" bitset); the actual A/C/G/T of an SNV
is materialised at the READ layer. Historically each modality invented that nucleotide identity
INDEPENDENTLY, so a single somatic SNV emitted CONTRADICTORY alleles — e.g. C->T in the DNA FASTQ
but T->C in the RNA/Visium FASTQ for the SAME locus — breaking iscc's "one tumour -> shared ground
truth across modalities" promise at the read level.

The fix: a single canonical per-locus (ref, alt) map (`iscc.data.dna.genome_bases`), a GENOME
PROPERTY (deterministic in the gene set + a FIXED genome seed, like `genome_features`), placed at
the variant site of every read reference. These tests are the regression that would have caught the
bug — strict equality of ref AND alt across DNA == RNA == Visium at every mutated locus.

CI-light: the scRNA/Visium FASTQs are self-contained (no external binary); the DNA per-cell FASTA is
always written (the DWGSIM/ART shell-out is the only binary-gated step, and is not needed here).
"""
import gzip
import os
from collections import defaultdict

import numpy as np
import pytest

from iscc.data import genome_bases
from iscc.data.dna import _segment_of, _TRANSITION, _TRANSVERSIONS
from iscc.data.reads import (
    SyntheticReference, RealGenomeReference, SyntheticTranscriptome,
    emit_reads, emit_scrna_reads, emit_visium_reads,
)
from iscc.tumor.models import GenotypeTumor


# --------------------------------------------------------------------------- helpers ----------
def _read_fasta_file(path):
    """Parse a FASTA file -> OrderedDict-like dict {record_name: sequence}."""
    recs, name, chunks = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    recs[name] = "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif name is not None:
                chunks.append(line.strip())
    if name is not None:
        recs[name] = "".join(chunks)
    return recs


def _read_fastq(path):
    with gzip.open(path, "rt") as fh:
        lines = fh.read().splitlines()
    return [(lines[i], lines[i + 1]) for i in range(0, len(lines), 4)]


def _seg_of_record(name):
    """`{cell}_seg{seg}_cp{cp}` -> seg (int)."""
    return int(name.split("_seg")[1].split("_")[0])


def _dna_observed_alleles(fasta_path, reference, mutated_loci):
    """Parse the emitted DNA FASTA: per mutated locus, (ref from the template, alt seen in reads).

    ref = the reference template base at the variant site; alt = the OTHER base observed across the
    per-copy records (the substituted somatic allele). alt is None if no copy carried the alt.
    """
    recs = _read_fasta_file(fasta_path)
    by_seg = defaultdict(list)
    for name, seq in recs.items():
        by_seg[_seg_of_record(name)].append(seq)
    out = {}
    for gl in mutated_loci:
        seg = next(s for s in reference.seg_ids if gl in reference.segments[s])
        pos = reference.locus_local_pos[gl]
        ref = reference.base_seq[seg][pos]
        seen = {seq[pos] for seq in by_seg.get(seg, [])}
        alts = seen - {ref}
        out[gl] = (ref, alts.pop() if alts else None)
    return out


def _rna_observed_alleles(fastq_r2_path, transcriptome):
    """Parse a (10x-style) R2 FASTQ: per gene, (ref at var_pos, alt base observed) — error-free."""
    vp = transcriptome.var_pos
    by_gene = defaultdict(set)
    for rid, seq in _read_fastq(fastq_r2_path):
        by_gene[rid.split(":")[1]].add(seq[vp])
    out = {}
    for g, seen in by_gene.items():
        ref = transcriptome.seq[g][vp]
        alts = seen - {ref}
        out[g] = (ref, alts.pop() if alts else None)
    return out


def _grow_tumor(seed=1, steps=500):
    GENOME = {"n_segments": 6, "segment_size": 40}
    SEL = {"prop_driver": 0.3, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
           "prop_treatment_resistance": 0.0, "driver_effects": 1.3, "dispersal_effects": 1.0,
           "treatment_resistant_effects": 1.0, "immune_resistant_effects": 1.0}
    C = {"division_rate": 0.4, "death_rate": 0.02, "max_birth_rate": 0.95,
         "mutation_rate": 0.6, "dispersal_rate": 0.2, "snv_prob": 0.7, "cnv_prob": 0.3}
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SEL,
                      cancer_cell_params=C, deme_params={"carrying_capacity": 6},
                      spatial_params={"grid_size": 14, "structure_radius": 0})
    t.grow(steps, seed=seed)
    return t


@pytest.fixture(scope="module")
def cell_data():
    return _grow_tumor().make_cell_data()


# ----------------------------------------------------------- genome_bases (the canonical map) --
class TestGenomeBases:
    GENES = [f"G_{s}_{p}" for s in range(6) for p in range(40)]

    def test_deterministic_in_seed(self):
        r1, a1 = genome_bases(self.GENES, seed=20240601)
        r2, a2 = genome_bases(self.GENES, seed=20240601)
        assert np.array_equal(r1, r2) and np.array_equal(a1, a2)        # same seed -> identical
        r3, a3 = genome_bases(self.GENES, seed=7)
        assert not (np.array_equal(r1, r3) and np.array_equal(a1, a3))  # different seed -> differs

    def test_ref_differs_from_alt_everywhere(self):
        ref, alt = genome_bases(self.GENES)
        assert np.all(ref != alt)                                       # alt is a real substitution
        assert set(ref).issubset(set("ACGT")) and set(alt).issubset(set("ACGT"))

    def test_order_and_subset_independent(self):
        """A locus's (ref, alt) depends only on its NAME — not its position or its neighbours."""
        ref, alt = genome_bases(self.GENES)
        idx = {g: i for i, g in enumerate(self.GENES)}
        shuffled = list(reversed(self.GENES))
        rs, as_ = genome_bases(shuffled)
        for j, g in enumerate(shuffled):
            assert rs[j] == ref[idx[g]] and as_[j] == alt[idx[g]]
        # a 3-gene subset reproduces those same loci exactly
        sub = ["G_2_10", "G_0_0", "G_5_39"]
        rsub, asub = genome_bases(sub)
        for k, g in enumerate(sub):
            assert rsub[k] == ref[idx[g]] and asub[k] == alt[idx[g]]

    def test_ti_tv_ratio_realistic(self):
        """The alt is drawn with a realistic ~2:1 transition:transversion ratio."""
        genes = [f"G_0_{p}" for p in range(4000)]
        ref, alt = genome_bases(genes, ti_tv=2.0)
        n_ti = sum(_TRANSITION[r] == a for r, a in zip(ref, alt))
        n_tv = sum(a in _TRANSVERSIONS[r] for r, a in zip(ref, alt))
        assert n_ti + n_tv == len(genes)                                # every alt is ti or tv
        assert 1.7 < n_ti / n_tv < 2.4                                  # ~2:1 as targeted


# ------------------------------------------------ references carry the canonical ref/alt -------
class TestReferencesCarryCanonical:
    GENES = [f"G_{s}_{p}" for s in range(4) for p in range(20)]

    def test_synthetic_reference_places_canonical_ref_and_alt(self):
        ref_b, alt_b = genome_bases(self.GENES)
        sr = SyntheticReference(self.GENES, seed=99, locus_length=50)   # run-seed differs from genome
        for gl, g in enumerate(self.GENES):
            seg = next(s for s in sr.seg_ids if gl in sr.segments[s])
            assert sr.base_seq[seg][sr.locus_local_pos[gl]] == ref_b[gl]  # canonical ref at the site
            assert sr.alt_base[gl] == alt_b[gl]                          # canonical alt exposed

    def test_synthetic_transcriptome_places_canonical_ref_and_alt(self):
        ref_b, alt_b = genome_bases(self.GENES)
        tx = SyntheticTranscriptome(self.GENES, seed=99, read_length=80)
        for gi, g in enumerate(self.GENES):
            assert tx.seq[g][tx.var_pos] == ref_b[gi]
            assert tx.alt_base[g] == alt_b[gi]

    def test_dna_and_rna_references_agree(self):
        """The two backends, built INDEPENDENTLY, agree on (ref, alt) at every locus (the fix)."""
        sr = SyntheticReference(self.GENES, seed=1)
        tx = SyntheticTranscriptome(self.GENES, seed=2)                 # different run seed
        for gl, g in enumerate(self.GENES):
            seg = next(s for s in sr.seg_ids if gl in sr.segments[s])
            assert sr.base_seq[seg][sr.locus_local_pos[gl]] == tx.seq[g][tx.var_pos]
            assert sr.alt_base[gl] == tx.alt_base[g]

    def test_real_genome_alt_consistent_with_rule(self, tmp_path):
        """The real-genome backend derives its alt from the FASTA ref by the SAME canonical rule."""
        fa = os.path.join(tmp_path, "g.fa")
        with open(fa, "w") as fh:
            for s in range(4):
                fh.write(f">chr{s}\n" + "ACGTACGTAC" * 30 + "\n")
        rg = RealGenomeReference(fa, self.GENES)
        for gl, g in enumerate(self.GENES):
            seg = next(s for s in rg.seg_ids if gl in rg.segments[s])
            ref = rg.base_seq[seg][rg.locus_local_pos[gl]]
            alt = rg.alt_base[gl]
            assert alt != ref                                           # a real substitution
            assert alt == _TRANSITION[ref] or alt in _TRANSVERSIONS[ref]


# ============================================================ THE CROSS-MODALITY GATE ==========
class TestCrossModalityAlleles:
    """One tumour -> emit DNA, scRNA and Visium reads -> the SAME ref AND alt at every mutated
    locus. This is the regression test for the independent-reference bug."""

    def test_map_level_every_mutated_locus_agrees(self, cell_data):
        """STRICT: at EVERY locus mutated anywhere in the tumour, the DNA reference, the scRNA/Visium
        transcriptome and the canonical map agree on both ref and alt — no expression dependence."""
        genes = list(cell_data["cell_snv"].columns)
        ref_b, alt_b = genome_bases(genes)
        sr = SyntheticReference(genes, seed=42)
        tx = SyntheticTranscriptome(genes, seed=42)        # Visium reuses SyntheticTranscriptome
        snv = cell_data["cell_snv"].values
        mutated = np.where((snv > 0).any(axis=0))[0]
        assert len(mutated) > 0
        for gl in mutated:
            g = genes[gl]
            seg = next(s for s in sr.seg_ids if gl in sr.segments[s])
            dna_ref = sr.base_seq[seg][sr.locus_local_pos[gl]]
            rna_ref = tx.seq[g][tx.var_pos]
            assert dna_ref == rna_ref == ref_b[gl]         # ref agrees: DNA == RNA(==Visium)
            assert sr.alt_base[gl] == tx.alt_base[g] == alt_b[gl]   # alt agrees: DNA == RNA(==Visium)

    def test_emitted_reads_agree_on_ref_and_alt(self, cell_data, tmp_path):
        """HEADLINE: parse the actually-emitted DNA FASTA + scRNA FASTQ + Visium FASTQ; at every
        locus where all three carry the alt, ref AND alt match. error-free so the alt is unambiguous.
        """
        genes = list(cell_data["cell_snv"].columns)
        gene_idx = {g: i for i, g in enumerate(genes)}

        # shared references (same genome_seed default) handed to each modality's emitter.
        dna_ref = SyntheticReference(genes, seed=42)
        tx = SyntheticTranscriptome(genes, seed=42, read_length=80)

        # DNA: bulk per-cell FASTA (always written; no binary needed for the template).
        snv = cell_data["cell_snv"].values
        mutated_loci = list(np.where((snv > 0).any(axis=0))[0])
        dna_res = emit_reads(cell_data, reference=dna_ref, modality="bulk", breadth="wgs",
                             seed=3, outdir=str(tmp_path / "dna"))
        dna = _dna_observed_alleles(dna_res["fasta"][0], dna_ref, mutated_loci)

        # scRNA + Visium: self-contained error-free FASTQ (error_rate=0 so alt is unambiguous).
        rna_res = emit_scrna_reads(cell_data, obs_fidelity=1.0, error_rate=0.0, seed=1,
                                   emit_fastq=True, transcriptome=tx, outdir=str(tmp_path / "rna"))
        vis_res = emit_visium_reads(cell_data, obs_fidelity=1.0, error_rate=0.0, seed=1,
                                    emit_fastq=True, transcriptome=tx, spot_pitch=2.0,
                                    spot_radius=0.6, outdir=str(tmp_path / "vis"))
        rna = _rna_observed_alleles(rna_res["fastq"][1], tx)
        vis = _rna_observed_alleles(vis_res["fastq"][1], tx)

        # loci where ALL THREE modalities actually emitted an alt-carrying read.
        checked = 0
        for gl in mutated_loci:
            g = genes[gl]
            d = dna.get(gl)
            r = rna.get(g)
            v = vis.get(g)
            if not (d and r and v):
                continue
            if d[1] is None or r[1] is None or v[1] is None:
                continue
            assert d[0] == r[0] == v[0], f"ref disagrees at {g}: DNA={d[0]} RNA={r[0]} Visium={v[0]}"
            assert d[1] == r[1] == v[1], f"alt disagrees at {g}: DNA={d[1]} RNA={r[1]} Visium={v[1]}"
            checked += 1
        assert checked > 0, "no mutated locus was emitted by all three modalities — gate vacuous"

    def test_independent_emitter_calls_yield_identical_map(self, cell_data, tmp_path):
        """Independent emitter invocations obtain the SAME canonical (ref, alt) per locus."""
        genes = list(cell_data["cell_snv"].columns)
        a = SyntheticTranscriptome(genes, seed=1)
        b = SyntheticTranscriptome(genes, seed=2, read_length=120)     # different run params
        assert all(a.alt_base[g] == b.alt_base[g] for g in genes)
        assert all(a.seq[g][a.var_pos] == b.seq[g][b.var_pos] for g in genes)
