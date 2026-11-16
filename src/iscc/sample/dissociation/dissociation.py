"""Dissociation biases: BIOLOGICAL sample-prep effects.

This models the sample-prep *biology* that happens when a solid tissue is
dissociated into a single-cell suspension — distinct from the assay's
sequencing-technical noise. Droplet doublets / ambient release are NOT modelled
here; those belong to the scRNA assay. This module is the sample-prep biology:

* **cell-type-dependent dissociation efficiency -> composition bias** (the
  headline piece): each biological type (cancer / immune / stromal /
  epithelial / host parenchyma) has a recovery probability, so the sampled type
  fractions differ from the true tissue fractions (e.g. immune is typically
  under-recovered by enzymatic dissociation).
* **dissociation-stress expression signature** (optional knob): a multiplicative
  shift on a small stress-gene set in ``cell_exp`` for dissociated cells.

  **How the stress genes are chosen.** Unless the caller names them (``stress_genes``), they are
  ``n_stress_genes`` genes (default 5% of the expression columns, at least one) drawn uniformly at
  random without replacement from ``cell_exp``'s columns, with the dissociation's own RNG and AFTER
  the recovery draw. So the set is not a curated stress programme: it is a random subset of
  whatever genes the expression table has, it differs between two dissociations with different
  RNGs (two patients, unless they share a seed or an explicit list), and the recovered cells do
  not depend on it. Every chosen gene of every recovered cell is multiplied by the same
  ``1 + stress_strength``, whatever the cell's type. :func:`choose_stress_genes` and
  :func:`apply_stress` expose the two steps, so a caller can fix one gene set and apply the
  signature at several strengths to the same recovered cells.

The recovery probabilities and the realized composition shift are returned for
``sample_meta.yaml``. Per-cell ground truth is preserved (only subset; the
stress signature, if enabled, returns a modified copy of ``cell_exp``).
"""
import numpy as np
import pandas as pd

from ...constants import normal_names

# Every NON-cancer cell type the engine can seed, taken from the engine's own list rather
# than repeated here, so no type is missed — e.g. ``host`` (the resident parenchyma of a
# metastatic deposit, i.e. normal tissue of the colonised organ), which would otherwise be
# counted as tumour and overstate the purity of any sample containing a metastasis.
NORMAL_TYPES = tuple(normal_names)

# Default per-type recovery probabilities. Immune cells are the hardest to
# recover from enzymatic tumour dissociation, epithelial the easiest — so the
# default biases composition AGAINST immune and toward epithelial/cancer.
# Host parenchyma dissociates like the other resident epithelium.
DEFAULT_RECOVERY = {
    "cancer": 0.9,
    "epithelial": 0.8,
    "host": 0.8,
    "stromal": 0.6,
    "immune": 0.4,
}


def biological_type(genotype_id):
    """Map a genotype id to its biological type.

    Non-cancer ids are the literal types ``immune``/``stromal``/``epithelial``/
    ``host``; any other id (a numeric clone id) is ``cancer``.
    """
    s = str(genotype_id)
    return s if s in NORMAL_TYPES else "cancer"


def choose_stress_genes(genes, n_stress_genes=None, rng=None):
    """The stress-gene set as :class:`Dissociation` draws it.

    ``n_stress_genes`` genes (default 5% of ``genes``, at least one, at most all) drawn uniformly
    without replacement with ``rng`` and returned in their column order. This is exactly the draw
    :class:`Dissociation` makes after its recovery draw, so passing a generator in the same state
    gives the same set.
    """
    genes = list(genes)
    n_genes = len(genes)
    if n_genes == 0:
        return []
    rng = rng if rng is not None else np.random.default_rng()
    n = n_stress_genes or max(1, int(round(0.05 * n_genes)))
    n = min(int(n), n_genes)
    gene_idx = rng.choice(n_genes, size=n, replace=False)
    return [genes[i] for i in np.sort(gene_idx)]


def apply_stress(exp, stress_strength, stress_genes, cells=None):
    """The dissociation-stress signature on an expression table: a copy of ``exp`` whose
    ``stress_genes`` columns are multiplied by ``1 + stress_strength`` in the rows ``cells``
    (default every row). ``stress_strength`` 0 returns an unchanged copy. Genes absent from
    ``exp`` are an error, so a gene list made for another gene set is not silently ignored."""
    stress_strength = float(stress_strength)
    if stress_strength < -1.0:
        raise ValueError(f"stress_strength must be >= -1 (a non-negative multiplier), got {stress_strength}")
    stress_genes = list(stress_genes)
    missing = [g for g in stress_genes if g not in exp.columns]
    if missing:
        raise ValueError(f"{len(missing)} stress gene(s) are not columns of the expression table, "
                         f"e.g. {missing[:3]}")
    out = exp.copy()
    if stress_strength == 0.0 or not stress_genes:
        return out
    rows = exp.index if cells is None else pd.Index(cells)
    out.loc[rows, stress_genes] *= (1.0 + stress_strength)
    return out


class Dissociation(object):
    """Apply cell-type-dependent recovery (+ optional stress signature).

    Parameters
    ----------
    cell_data : dict[str, pandas.DataFrame]
        The (already biopsied) ground truth; must include ``cell_type``.
    rng : numpy.random.Generator, optional
    recovery_probs : dict[str, float], optional
        Per-biological-type recovery probability (defaults to DEFAULT_RECOVERY).
    stress_strength : float
        Multiplicative bump (1 + stress_strength) applied to the stress-gene set
        of recovered cells' expression. 0 disables the signature.
    n_stress_genes : int, optional
        Size of the stress-gene set (default ~5% of genes, >=1).
    stress_genes : list of str, optional
        The stress-gene set itself, replacing the random draw (see the module notes); every gene
        must be a column of ``cell_exp``. ``n_stress_genes`` is then ignored and no gene draw is
        made, so the recovered cells are the same either way.
    """

    def __init__(self, cell_data, rng=None, recovery_probs=None,
                 stress_strength=0.0, n_stress_genes=None, stress_genes=None):
        self.cell_data = cell_data
        self.rng = rng if rng is not None else np.random.default_rng()
        self.recovery_probs = dict(DEFAULT_RECOVERY)
        if recovery_probs:
            self.recovery_probs.update(recovery_probs)
        self.stress_strength = float(stress_strength)
        self.n_stress_genes = n_stress_genes
        self.stress_genes = None if stress_genes is None else list(stress_genes)

    def run(self):
        """Returns (chosen_index, meta, exp_override).

        ``exp_override`` is a modified copy of ``cell_exp`` when the stress
        signature is enabled, else None (caller writes the original).
        """
        types = self.cell_data["cell_type"]["cell_id"]
        bio = types.map(biological_type)
        probs = bio.map(lambda t: self.recovery_probs.get(t, 1.0)).to_numpy(dtype=float)
        keep = self.rng.random(len(probs)) < probs
        chosen = types.index[keep]

        input_comp = bio.value_counts(normalize=True).to_dict()
        sampled_comp = bio[keep].value_counts(normalize=True).to_dict()
        shift = {t: float(sampled_comp.get(t, 0.0) - input_comp.get(t, 0.0))
                 for t in input_comp}

        exp_override, stress_genes = self._apply_stress(keep)

        meta = dict(
            recovery_probs={t: float(p) for t, p in self.recovery_probs.items()},
            input_composition={t: float(v) for t, v in input_comp.items()},
            sampled_composition={t: float(v) for t, v in sampled_comp.items()},
            composition_shift=shift,
            stress_strength=self.stress_strength,
            stress_genes=stress_genes,
            stress_genes_source=("given" if self.stress_genes is not None else "random"),
            n_input=int(len(probs)),
            n_recovered=int(keep.sum()),
        )
        return chosen, meta, exp_override

    def _apply_stress(self, keep):
        """Apply the dissociation-stress signature to recovered cells' exp."""
        exp = self.cell_data.get("cell_exp")
        if exp is None or self.stress_strength == 0.0:
            return None, []
        if self.stress_genes is not None:
            stress_genes = list(self.stress_genes)
        else:
            stress_genes = choose_stress_genes(exp.columns, self.n_stress_genes, self.rng)
        # Multiply the stress-gene columns for recovered cells only.
        return apply_stress(exp, self.stress_strength, stress_genes, cells=exp.index[keep]), stress_genes
