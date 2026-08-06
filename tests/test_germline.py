"""Germline (inherited) variants + the truncal layer of the passenger overlay.

Two ground-truth changes a DNA analysis needs, tested together because they are the two halves of a
readable variant-allele-fraction spectrum:

  * GERMLINE — an inherited background carried by EVERY cell of the patient, tumour and normal alike.
    Heterozygous sites read at 0.5 and homozygous ones at 1.0 in unaltered tissue, which is the anchor
    a purity / cancer-cell-fraction analysis calibrates against. Off by default, and provably inert:
    with ``germline_params=None`` the engine is indistinguishable from one built before the layer
    existed.
  * TRUNCAL passengers — the clade containing every sampled cancer cell now gets its own marker
    budget, so a share of loci is carried by ALL of them. Without it no locus reached a cancer-cell
    fraction above ~0.34 and a bulk histogram had no clonal peak at all.
"""
import os
import sys

import numpy as np
import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "validation"))
import realistic_regime as RR  # noqa: E402

from iscc.tumor.models import GenotypeTumor  # noqa: E402
from iscc.tumor.germline import draw_germline_sites, apply_germline, neutral_positions  # noqa: E402

# Enough drawn sites for the statistics below to mean something on a 600-gene toy genome.
GERMLINE = {"het_frac": 0.05, "hom_frac": 0.33, "seed": 11}

GENOME = {"n_segments": 4, "segment_size": 150}
SELECTION = {"prop_driver": 0.05, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
             "prop_treatment_resistance": 0.0, "prop_met_survival": 0.0}
# Low CNA rate: the point of these tests is the germline INJECTION, and copy-number loss of
# heterozygosity would otherwise delete single-homolog variants out from under the assertions.
CANCER = {"division_rate": 0.6, "death_rate": 0.05, "max_birth_rate": 0.95,
          "mutation_rate": 0.4, "n_snvs_per_allele": 0.1, "dispersal_rate": 0.4,
          "snv_prob": 0.97, "cnv_prob": 0.03}
DEME = {"carrying_capacity": 10, "initial_cancer_cells": 5}
SPATIAL = {"grid_size": 15, "structure_radius": 4}       # seeds epithelial + stromal normal cells


def _tumor(seed=1, steps=60, germline=None, coarsen=False, **kw):
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
                      germline_params=germline, coarsen_passengers=coarsen, **kw)
    t.grow(n_steps=steps, seed=seed)
    return t


@pytest.fixture(scope="module")
def germline_tumor():
    return _tumor(germline=GERMLINE)


def _celltypes(t):
    ids = t.cell_data["cell_type"]["cell_id"].astype(str).values
    return np.array([t.genotypes[g].type for g in ids])


# =========================================================================== off by default
def test_germline_off_is_byte_identical():
    """``germline_params=None`` must be indistinguishable from a build that never mentions germline:
    same growth, same per-cell ground truth, no extra frames or annotation keys."""
    absent = _tumor(seed=4)
    explicit = _tumor(seed=4, germline=None)
    assert absent.get_tumor_size() == explicit.get_tumor_size()
    for key in ("cell_snv", "cell_cnv", "cell_exp", "cell_crd"):
        assert np.array_equal(absent.cell_data[key].values, explicit.cell_data[key].values), key
    for t in (absent, explicit):
        assert len(t.germline_sites) == 0
        assert "germline_types" not in t.get_gene_data()
        assert not np.any(t.selection.germline_types)


def test_empty_germline_params_stays_off():
    """An empty dict is 'no germline', not 'germline with defaults' — nothing is drawn."""
    t = GenotypeTumor(seed=4, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL,
                      germline_params={})
    assert len(t.germline_sites) == 0


# =========================================================================== the draw
def test_sites_are_neutral_and_zygosity_split_is_as_asked():
    sel = GenotypeTumor(seed=1, genome_params=GENOME, selection_params=SELECTION,
                        cancer_cell_params=CANCER, deme_params=DEME,
                        spatial_params=SPATIAL).selection
    sites, zyg, hom = draw_germline_sites(sel, **GERMLINE)
    assert len(sites) == len(zyg) == len(hom) > 10
    assert len(set(sites.tolist())) == len(sites)                 # distinct loci
    assert np.all(np.diff(sites) > 0)                             # returned in ascending order
    # never on a gene with any selective role -> the germline cannot perturb fitness
    functional = set(range(sel.n_genes)) - set(neutral_positions(sel).tolist())
    assert not (set(sites.tolist()) & functional)
    # hom_frac is the fraction of the DRAWN sites that are homozygous
    assert abs((zyg == "hom").mean() - GERMLINE["hom_frac"]) < 0.05
    assert set(hom[zyg == "hom"]) == {"both"}
    assert set(hom[zyg == "het"]) <= {"p", "m"}


def test_heterozygous_homolog_is_not_one_sided():
    """A het variant sits on ONE parental copy, picked at random per site. Always using the same
    homolog would make the germline B-allele track perfectly one-sided."""
    sel = GenotypeTumor(seed=1, genome_params=GENOME, selection_params=SELECTION,
                        cancer_cell_params=CANCER, deme_params=DEME,
                        spatial_params=SPATIAL).selection
    _, zyg, hom = draw_germline_sites(sel, het_frac=0.2, hom_frac=0.33, seed=3)
    het = hom[zyg == "het"]
    assert (het == "p").sum() > 0 and (het == "m").sum() > 0
    assert 0.3 < (het == "p").mean() < 0.7


def test_draw_is_reproducible_and_private_per_seed():
    sel = GenotypeTumor(seed=1, genome_params=GENOME, selection_params=SELECTION,
                        cancer_cell_params=CANCER, deme_params=DEME,
                        spatial_params=SPATIAL).selection
    a = draw_germline_sites(sel, **GERMLINE)
    b = draw_germline_sites(sel, **GERMLINE)
    c = draw_germline_sites(sel, **{**GERMLINE, "seed": 12})
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])
    assert not np.array_equal(a[0], c[0])          # a different patient, a different background


# =========================================================================== what the cells read
def test_normal_cells_read_half_at_het_and_one_at_hom(germline_tumor):
    """The headline for a purity anchor: in unaltered (diploid, static) tissue a heterozygous germline
    site reads at exactly VAF 0.5 and a homozygous one at exactly 1.0 — in EVERY normal cell."""
    t = germline_tumor
    types = _celltypes(t)
    normal = np.isin(types, ["epithelial", "stromal", "immune", "host"])
    assert normal.sum() > 0 and (types == "cancer").sum() > 0
    snv = t.cell_data["cell_snv"].values[normal]
    het = t.germline_sites[t.germline_zygosity == "het"]
    hom = t.germline_sites[t.germline_zygosity == "hom"]
    assert len(het) > 0 and len(hom) > 0
    np.testing.assert_allclose(snv[:, het], 0.5, atol=1e-12)
    np.testing.assert_allclose(snv[:, hom], 1.0, atol=1e-12)
    # and nothing ELSE is mutated in a normal cell: the germline is their whole variant content
    other = np.setdiff1d(np.arange(t.n_genes), t.germline_sites)
    assert snv[:, other].max() == 0.0


def test_germline_is_carried_by_the_tumour_too(germline_tumor):
    """Inherited means inherited by every cell of the individual — the cancer clones carry it as
    well, which is why it is the background a tumour sample's purity is read against."""
    t = germline_tumor
    snv = t.cell_data["cell_snv"].values[_celltypes(t) == "cancer"]
    hom = t.germline_sites[t.germline_zygosity == "hom"]
    het = t.germline_sites[t.germline_zygosity == "het"]
    # A homozygous site is on EVERY copy, so it stays at 1.0 whatever the copy number does: an
    # amplification duplicates a variant copy and a deletion removes one. That the value survives
    # copy-number churn is the check that the germline really sits under the somatic events rather
    # than being painted on afterwards.
    np.testing.assert_allclose(snv[:, hom], 1.0, atol=1e-12)
    assert (snv[:, het] > 0).mean() > 0.9               # single-copy variants, a few lost to deletion


def test_germline_does_not_perturb_fitness():
    """Drawn from neutral positions only, so the founder's evolved rates are exactly what they are
    without a germline — the layer is observational, never selective."""
    off = _tumor(seed=4, steps=0)
    on = _tumor(seed=4, steps=0, germline=GERMLINE)
    assert on.germline_sites.size > 0
    assert (off.genotypes[off.founder_id].evolutionary_parameters
            == on.genotypes[on.founder_id].evolutionary_parameters)
    for key in ("n_mut_onc", "n_mut_tsg", "n_mutated_drivers", "ploidy", "seg_cns"):
        assert (off.genotypes[off.founder_id].genome_summary[key]
                == on.genotypes[on.founder_id].genome_summary[key]), key


def test_germline_lands_on_the_recorded_homolog():
    """The allele-resolved frames put a het variant on exactly the parental copy the ground truth
    names — so a B-allele-frequency readout is consistent with the recorded truth."""
    t = _tumor(seed=2, germline=GERMLINE, coarsen=True)
    cd = t.cell_data
    assert "cell_snv_p" in cd
    normal = np.isin(_celltypes(t), ["epithelial", "stromal", "immune", "host"])
    p, m = cd["cell_snv_p"].values[normal], cd["cell_snv_m"].values[normal]
    het = t.germline_zygosity == "het"
    on_p = t.germline_sites[het & (t.germline_homolog == "p")]
    on_m = t.germline_sites[het & (t.germline_homolog == "m")]
    np.testing.assert_allclose(p[:, on_p], 0.5, atol=1e-12)
    np.testing.assert_allclose(m[:, on_p], 0.0, atol=1e-12)
    np.testing.assert_allclose(m[:, on_m], 0.5, atol=1e-12)
    np.testing.assert_allclose(p[:, on_m], 0.0, atol=1e-12)


# =========================================================================== ground truth round-trip
def test_ground_truth_masks_round_trip(germline_tumor, tmp_path):
    """The per-site truth on the tumour and the per-gene mask in the gene annotation agree, and the
    mask survives a write to the canonical layout."""
    import pandas as pd
    t = germline_tumor
    n = len(t.germline_sites)
    assert n > 0 and len(t.germline_zygosity) == n and len(t.germline_homolog) == n
    gd = t.get_gene_data()
    assert "germline_types" in gd
    mask = gd["germline_types"].values.ravel()
    assert mask.shape[0] == t.n_genes
    assert set(np.flatnonzero(mask == 1)) == set(t.germline_sites[t.germline_zygosity == "het"])
    assert set(np.flatnonzero(mask == 2)) == set(t.germline_sites[t.germline_zygosity == "hom"])
    assert int((mask != 0).sum()) == n

    out = tmp_path / "run"
    t.write(out)
    back = pd.read_csv(out / "gene_data" / "germline_types.csv", index_col=0).values.ravel()
    assert np.array_equal(back, mask)


def test_config_can_enable_germline(tmp_path):
    """A YAML config turns the layer on, exactly like update_mode / max_cells."""
    cfg = {"genome_params": GENOME, "selection_params": SELECTION,
           "deme_params": DEME, "spatial_params": SPATIAL,
           "cell_params": {"cancer": CANCER},
           "germline_params": GERMLINE}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    t = GenotypeTumor(config=str(path), seed=1)
    assert len(t.germline_sites) > 0
    assert t.germline_params == GERMLINE


def test_apply_germline_can_be_called_on_a_built_tumour():
    """The two calls are usable directly (the cohort layer seeds its private per-patient markers this
    way), and applying them refreshes the per-deme event rates."""
    t = GenotypeTumor(seed=1, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=CANCER, deme_params=DEME, spatial_params=SPATIAL)
    assert len(t.germline_sites) == 0
    sites, zyg, hom = draw_germline_sites(t.selection, **GERMLINE)
    apply_germline(t, sites, zyg, hom)
    assert np.array_equal(t.germline_sites, sites)
    assert t.deme_rates.shape[0] == len(t.demes)
    founder = t.genotypes[t.founder_id]
    assert (founder.get_snvs()[sites] > 0).all()


# =========================================================================== the truncal layer
@pytest.fixture(scope="module")
def clonal_tumor():
    """The same small realistic tumour the coalescent-overlay tests use (coarsening ON, so the
    passenger overlay actually runs)."""
    return RR.grow_realistic(seed=3, target_cancer=3000, scale="small", coarsen=True,
                             cancer={"mutation_rate": 0.6, "n_snvs_per_allele": 0.1, "wgd_rate": 0.1},
                             materialise=True, max_cells=4000)


def _cancer_frames(t):
    ids = t.cell_data["cell_type"]["cell_id"].astype(str).values
    is_can = np.array([t._is_cancer(g) for g in ids])
    return (t.cell_data["cell_snv"].values[is_can],
            t.cell_data["cell_cnv"].values[is_can].astype(float))


def test_truncal_layer_gives_a_clonal_ccf_mode(clonal_tumor):
    """Cancer-cell fraction (VAF x copy number / purity, purity 1 on a cancer-only sample) now reaches
    ~1.0 at a whole set of loci — the clonal peak. Before the truncal layer the maximum over all loci
    was ~0.05 at this scale and ~0.34 at a larger one: no locus was carried by every cancer cell."""
    snv, cnv = _cancer_frames(clonal_tumor)
    vaf = (snv * cnv).sum(0) / np.maximum(cnv.sum(0), 1e-9)
    ccf = vaf * cnv.mean(0)
    assert ccf.max() >= 0.95
    assert (ccf >= 0.9).sum() >= 20, f"expected a clonal MODE, not a spike: {(ccf >= 0.9).sum()} loci"
    # and those loci really are in every cancer cell, not an averaging artefact
    carrier = (snv > 1e-9).mean(0)
    assert carrier.max() > 0.99
    assert (carrier > 0.99).sum() >= 20


def test_truncal_layer_keeps_the_subclonal_tail(clonal_tumor):
    """A clonal peak must not swallow the spectrum: most mutated loci are still subclonal, so the
    site-frequency spectrum keeps its tail."""
    snv, _ = _cancer_frames(clonal_tumor)
    carrier = (snv > 1e-9).mean(0)
    mutated = carrier > 0
    assert mutated.sum() > 100
    assert (carrier[mutated] < 0.5).mean() > 0.5


def test_allele_invariant_holds_with_the_truncal_layer(clonal_tumor):
    """The overlay's contract is unchanged: the two homolog frames sum EXACTLY to the collapsed one
    and no variant allele fraction exceeds 1."""
    cd = clonal_tumor.cell_data
    p, m, s = cd["cell_snv_p"].values, cd["cell_snv_m"].values, cd["cell_snv"].values
    assert np.abs(p + m - s).max() == 0.0
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_truncal_markers_are_inherited_by_every_clone(clonal_tumor):
    """Truncal markers are born on the trunk and descend to the whole tumour, so they are shared BY
    CONSTRUCTION rather than by coincidence: every materialised cancer clone carries essentially all
    of them. Not literally all — a copy-number deletion can still remove a truncal variant from the
    clone that suffered it, which is the real behaviour and the reason a clonal peak has a shoulder."""
    t = clonal_tumor
    ids = t.cell_data["cell_type"]["cell_id"].astype(str).values
    is_can = np.array([t._is_cancer(g) for g in ids])
    can_ids = ids[is_can]
    snv = t.cell_data["cell_snv"].values[is_can]
    clonal = np.flatnonzero((snv > 1e-9).mean(0) > 0.99)
    assert len(clonal) >= 20
    kept = [(snv[can_ids == gid][:, clonal] > 1e-9).mean() for gid in sorted(set(can_ids))]
    assert min(kept) > 0.8, f"a clone lost most of the truncal layer (min kept {min(kept):.2f})"
    assert float(np.mean(kept)) > 0.98
