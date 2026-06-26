"""Real-genome specification for the fit-to-real / Charm milestone (DESIGN_inference A.5).

iscc's abstract genome (synthetic segments, random driver layout, a single scalar
``driver_effects``) is replaced here by a **human chromosome-arm genome**: the ~39 autosomal p/q
arms become the engine's segments, each sized in proportion to the arm's length, and selection
acts through a **per-arm copy-number coefficient vector** ``s_arm`` (CINner's arm model) rather
than the scalar driver effect. The engine itself is unchanged apart from the per-arm selection
mode added to :class:`iscc.tumor.components.selection.Selection`; this module only assembles the
configuration it consumes.

A :class:`GenomeSpec` carries, per arm:
  * ``arm_names`` — e.g. ``"8q"`` (chromosome + p/q),
  * ``arm_lengths`` — base pairs (UCSC cytoBand), mapped to per-arm ``segment_sizes``,
  * ``onc_counts`` / ``tsg_counts`` — oncogene / tumour-suppressor content (COSMIC CGC + Davoli),
  * ``charm`` — Davoli 2013 "Charm" net-selection score (orthogonal validation target),
  * ``s_arm`` — the per-arm selection vector actually used by the engine (defaults to neutral
    1.0; the ABC layer infers it, and :meth:`prior_from_content` gives a content-informed prior).

It is built from the derived CSVs under ``validation/data/`` (themselves produced + cited by
``validation/data/build_realgenome_data.py``), via :func:`load_default`.

**v1 scope (honest):** SNV drivers are off (``prop_driver=0``) — the arm-CN model is what fits
PCAWG; per-arm oncogene/TSG counts inform the ``s_arm`` *prior* and the Charm comparison but do
not place gene drivers in the engine (the arm-CN fitness reads only ``seg_cns``). Arm length sets
the per-arm gene grid (``segment_sizes``), so genome-fraction summaries are arm-length-weighted.
"""
import os

import numpy as np


class GenomeSpec:
    def __init__(self, arm_names, arm_lengths, onc_counts, tsg_counts,
                 charm=None, s_arm=None, gene_scale=None, min_genes=4, max_genes=60):
        self.arm_names = list(arm_names)
        self.n_arms = len(self.arm_names)
        self.arm_lengths = np.asarray(arm_lengths, dtype=float)
        self.onc_counts = np.asarray(onc_counts, dtype=float)
        self.tsg_counts = np.asarray(tsg_counts, dtype=float)
        self.charm = None if charm is None else np.asarray(charm, dtype=float)
        self.s_arm = None if s_arm is None else np.asarray(s_arm, dtype=float)

        # Map arm length -> a (small) per-arm gene grid. The grid is kept modest so inference
        # sims stay cheap (the engine does one event per update); arm length enters as the
        # *relative* size between arms, clamped to [min_genes, max_genes].
        lengths = self.arm_lengths
        frac = lengths / lengths.max() if lengths.max() > 0 else np.ones_like(lengths)
        sizes = np.round(min_genes + frac * (max_genes - min_genes)).astype(int)
        self.segment_sizes = np.clip(sizes, min_genes, max_genes).astype(int).tolist()

    # --- engine wiring -------------------------------------------------------
    def engine_params(self, selection_params=None):
        """Build ``(genome_params, selection_params)`` for ``GenotypeTumor(genome_mode='real')``.

        Any keys in the caller's ``selection_params`` win (so the ABC layer can inject the
        ``s_arm`` vector it is inferring); the rest default to the per-arm CN model with SNV
        drivers off.
        """
        sp = dict(selection_params or {})
        sp.setdefault("selection_mode", "arm")
        sp.setdefault("prop_driver", 0.0)
        sp.setdefault("prop_dispersal", 0.0)
        sp.setdefault("prop_immune_resistance", 0.0)
        sp.setdefault("prop_treatment_resistance", 0.0)
        if "s_arm" not in sp:
            sp["s_arm"] = (self.s_arm.tolist() if self.s_arm is not None
                           else np.ones(self.n_arms).tolist())
        gp = {"n_segments": self.n_arms,
              "segment_sizes": list(self.segment_sizes),
              "segment_size": int(round(float(np.mean(self.segment_sizes))))}
        return gp, sp

    # --- priors / content ----------------------------------------------------
    def content_score(self):
        """Net oncogenic content per arm = (#oncogenes - #TSGs), the sign of expected selection.

        Positive -> amplification expected to be beneficial (``s_arm > 1``); negative -> deletion
        beneficial (``s_arm < 1``). Used to centre a content-informed prior and as a sanity axis.
        """
        return self.onc_counts - self.tsg_counts

    def with_s_arm(self, s_arm):
        """Return a copy of this spec with a concrete ``s_arm`` (e.g. a fitted MAP)."""
        return GenomeSpec(self.arm_names, self.arm_lengths, self.onc_counts, self.tsg_counts,
                          charm=self.charm, s_arm=np.asarray(s_arm, dtype=float),
                          min_genes=min(self.segment_sizes), max_genes=max(self.segment_sizes))


def load_default(data_dir=None):
    """Load the default human-arm :class:`GenomeSpec` from ``validation/data/realgenome_arms.csv``.

    The CSV is produced by ``validation/data/build_realgenome_data.py`` (download + provenance).
    Columns: ``arm, length_bp, n_onc, n_tsg, charm`` (charm may be blank for arms without a score).
    """
    import csv
    if data_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(here, "..", "..", "..", "validation", "data")
    path = os.path.join(data_dir, "realgenome_arms.csv")
    names, lengths, onc, tsg, charm = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            names.append(row["arm"])
            lengths.append(float(row["length_bp"]))
            onc.append(float(row["n_onc"]))
            tsg.append(float(row["n_tsg"]))
            charm.append(float(row["charm"]) if row.get("charm") not in (None, "", "NA") else np.nan)
    return GenomeSpec(names, lengths, onc, tsg, charm=np.asarray(charm, dtype=float))


def load_real_cna_profile(data_dir=None, cancer_type=None):
    """Load the observed per-arm CNA gain/loss frequencies (fit-to-real target).

    Returns ``(arm_names, gain_freq, loss_freq)`` aligned to ``realgenome_cna.csv`` row order.
    ``cancer_type`` selects a column group when the file holds several cancer types; ``None``
    uses the default (pan-cancer) columns.
    """
    import csv
    if data_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(here, "..", "..", "..", "validation", "data")
    path = os.path.join(data_dir, "realgenome_cna.csv")
    gpref = f"gain_{cancer_type}" if cancer_type else "gain"
    lpref = f"loss_{cancer_type}" if cancer_type else "loss"
    names, gain, loss = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            names.append(row["arm"])
            gain.append(float(row[gpref]))
            loss.append(float(row[lpref]))
    return names, np.asarray(gain, dtype=float), np.asarray(loss, dtype=float)
