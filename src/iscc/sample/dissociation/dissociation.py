"""F2 — dissociation biases: BIOLOGICAL sample-prep effects.

This models the sample-prep *biology* that happens when a solid tissue is
dissociated into a single-cell suspension — distinct from the assay's
sequencing-technical noise. Droplet doublets / ambient release are NOT modelled
here; those belong to the F3 assay. F2 stays the sample-prep biology:

* **cell-type-dependent dissociation efficiency -> composition bias** (the
  headline piece): each biological type (cancer / immune / stromal /
  epithelial / host parenchyma) has a recovery probability, so the sampled type
  fractions differ from the true tissue fractions (e.g. immune is typically
  under-recovered by enzymatic dissociation).
* **dissociation-stress expression signature** (optional knob): a multiplicative
  shift on a small stress-gene set in ``cell_exp`` for dissociated cells.

The recovery probabilities and the realized composition shift are returned for
``sample_meta.yaml``. Per-cell ground truth is preserved (only subset; the
stress signature, if enabled, returns a modified copy of ``cell_exp``).
"""
import numpy as np

from ...constants import normal_names

# Every NON-cancer cell type the engine can seed, taken from the engine's own list rather
# than repeated here — a hand-written copy goes stale the moment a type is added, which is
# exactly what happened to ``host`` (the resident parenchyma of a metastatic deposit, i.e.
# normal tissue of the colonised organ). While it was missing, every host cell was counted
# as a tumour cell, overstating the purity of any sample containing a metastasis.
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
    """

    def __init__(self, cell_data, rng=None, recovery_probs=None,
                 stress_strength=0.0, n_stress_genes=None):
        self.cell_data = cell_data
        self.rng = rng if rng is not None else np.random.default_rng()
        self.recovery_probs = dict(DEFAULT_RECOVERY)
        if recovery_probs:
            self.recovery_probs.update(recovery_probs)
        self.stress_strength = float(stress_strength)
        self.n_stress_genes = n_stress_genes

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
            n_input=int(len(probs)),
            n_recovered=int(keep.sum()),
        )
        return chosen, meta, exp_override

    def _apply_stress(self, keep):
        """Apply the dissociation-stress signature to recovered cells' exp."""
        exp = self.cell_data.get("cell_exp")
        if exp is None or self.stress_strength == 0.0:
            return None, []
        n_genes = exp.shape[1]
        n = self.n_stress_genes or max(1, int(round(0.05 * n_genes)))
        n = min(n, n_genes)
        gene_idx = self.rng.choice(n_genes, size=n, replace=False)
        stress_genes = list(exp.columns[np.sort(gene_idx)])
        out = exp.copy()
        # Multiply the stress-gene columns for recovered cells only.
        recovered_idx = exp.index[keep]
        out.loc[recovered_idx, stress_genes] *= (1.0 + self.stress_strength)
        return out, stress_genes
