"""iscc.integrations.cci — F8's OWN ligand-receptor database, and the CCI export seam.

W0 (DESIGN_cci_spatial.md). iscc emits its own L-R database over its own abstract gene
identifiers (``G_1_10``, ``G_3_30``, …) rather than naming genes so a curated human database can
match them. The analysis tools (CellChat first) are then pointed at THIS database, so gene
identifiers stay abstract and no real cancer-gene annotation is imported.

The database and F8's wiring are two outputs of ONE object: a pair is *active* precisely because
F8's CCI channel is wired to it (``tumor.microenv_truth["cci_wired_pair"]``). Everything else in the
database is an unwired decoy. The pair CLASSES are MEASURED, not planted:

  * **active**  — the wired pair, ``row 0`` of ``cci_pairs`` (known by construction);
  * **candidate** — every other pair (a neutral decoy needs no engineering);
  * **clone-correlated** — computed per candidate afterwards by :func:`clone_correlation`, as how
    much of its ligand/receptor expression variance clone identity explains. This is W4's analysis
    axis, an OBSERVATION on the generated data, not a knob.

The writer targets the exact ``Update-CellChatDB`` tutorial shape verified in
``validation/README_cellchat.md``: strict 1:1 pairs, empty ``complex``/``cofactor`` tables, and a
``geneInfo`` whitelist that MUST list every gene the interaction table references (the single silent
point of failure, §4.1). ``write_cci_database`` asserts that whitelist is complete before writing.

Additive and dependency-free: nothing here changes the engine; pandas is already a core dependency.
"""
import os

import numpy as np
import pandas as pd

# The four legal CellChat `annotation` levels (README_cellchat.md §4.2). CellChat `factor()`s the
# column against these, so anything else silently becomes NA and corrupts the diffusive-vs-contact
# split. The wired CCI field is a diffusive (secreted) niche signal.
ANNOTATION_SECRETED = "Secreted Signaling"
_LEGAL_ANNOTATIONS = ("Secreted Signaling", "ECM-Receptor", "Non-protein Signaling",
                      "Cell-Cell Contact")

# CellChat interaction columns, in the tutorial order (README_cellchat.md §7). `wired` is an extra
# passthrough column iscc adds (CellChat preserves unknown columns, §4.2) so a scorer can tell the
# active pair from the decoys without a side file.
_INTERACTION_COLUMNS = ["interaction_name", "pathway_name", "ligand", "receptor",
                        "agonist", "antagonist", "co_A_receptor", "co_I_receptor",
                        "annotation", "interaction_name_2", "wired"]


def cci_database(tumor, annotation=ANNOTATION_SECRETED):
    """Build iscc's own L-R database from a grown tumour's F8 wiring + candidate pairs.

    ``tumor.make_cell_data()`` must have run (so ``microenv_truth`` carries ``cci_pairs`` and
    ``cci_wired_pair``). Returns a dict of four DataFrames ``{interaction, complex, cofactor,
    geneInfo}`` in the ``Update-CellChatDB`` tutorial shape.

    Parameters
    ----------
    tumor : GenotypeTumor
        A tumour grown with an F8 CCI channel active (``microenv_params['cci']`` with non-zero
        ``strength`` and ``n_target_genes``). ``n_candidate_pairs`` sets how many decoys accompany
        the wired pair.
    annotation : str
        One of the four legal CellChat annotation levels; defaults to ``"Secreted Signaling"``
        (the wired CCI field is a diffusive secreted signal).

    Returns
    -------
    dict of DataFrame
        ``interaction`` (one row per pair, ``wired`` flags the active one), ``geneInfo`` (the
        ``Symbol`` whitelist of every referenced gene), and empty ``complex`` / ``cofactor``
        (0 rows × 2 columns — the round-trip-safe shape; never 1 column, README §8.1).
    """
    if annotation not in _LEGAL_ANNOTATIONS:
        raise ValueError(f"annotation must be one of {_LEGAL_ANNOTATIONS}, got {annotation!r}")
    truth = getattr(tumor, "microenv_truth", None)
    if not truth or "cci_pairs" not in truth:
        raise ValueError("tumor has no CCI database — grow it with an active F8 cci channel and "
                         "call make_cell_data() first (microenv_params['cci'] with strength>0).")
    pairs = np.asarray(truth["cci_pairs"], dtype=int)
    if pairs.size == 0:
        raise ValueError("the CCI database is empty — the cci channel drew no wired pair "
                         "(needs strength != 0 and n_target_genes > 0).")
    wired_idx = int(truth.get("cci_wired_pair", 0))
    gene_names = tumor.selection.get_gene_names()

    rows = []
    for i, (lig_i, rec_i) in enumerate(pairs):
        lig, rec = gene_names[int(lig_i)], gene_names[int(rec_i)]
        rows.append({
            "interaction_name": f"{lig}_{rec}",
            "pathway_name": f"CCI_{i:03d}",              # unique per pair -> per-pair prob is direct
            "ligand": lig, "receptor": rec,
            "agonist": "", "antagonist": "", "co_A_receptor": "", "co_I_receptor": "",
            "annotation": annotation,
            "interaction_name_2": f"{lig} - {rec}",
            "wired": bool(i == wired_idx),
        })
    interaction = pd.DataFrame(rows, columns=_INTERACTION_COLUMNS)
    interaction.index = interaction["interaction_name"]

    genes = sorted(set(interaction["ligand"]) | set(interaction["receptor"]))
    gene_info = pd.DataFrame({"Symbol": genes})
    # Empty complex/cofactor with TWO columns (0×2). A single-column table crashes CellChat with a
    # misleading error (README §8.1); 0×0 or ≥2 columns are safe, and 0×2 survives the CSV round trip.
    complex_df = pd.DataFrame(columns=["subunit_1", "subunit_2"])
    cofactor_df = pd.DataFrame(columns=["cofactor1", "cofactor2"])
    return {"interaction": interaction, "complex": complex_df,
            "cofactor": cofactor_df, "geneInfo": gene_info}


def referenced_genes(db):
    """The set of gene symbols the interaction (and complex) tables reference — the whitelist that
    ``geneInfo$Symbol`` must equal, or CellChat silently drops the missing pairs (README §4.1)."""
    inter = db["interaction"]
    genes = set(inter["ligand"]) | set(inter["receptor"])
    cx = db.get("complex")
    if cx is not None and len(cx):
        genes |= set(np.asarray(cx.values).ravel().tolist())
    return {g for g in genes if isinstance(g, str) and g}


def write_cci_database(tumor_or_db, out_dir, annotation=ANNOTATION_SECRETED):
    """Write the L-R database as the four ``Update-CellChatDB`` CSVs and return their paths.

    Accepts a grown ``tumor`` (builds the database first) or a pre-built ``db`` dict. Before
    writing, ASSERTS ``geneInfo$Symbol`` covers every referenced gene — the one assertion that
    catches the entire silent-drop failure class (README §4.1, §7). Returns
    ``(paths, expected_genes)`` so an R runner can re-assert ``setequal(extractGene(db), expected)``.
    """
    db = tumor_or_db if isinstance(tumor_or_db, dict) else cci_database(tumor_or_db, annotation)
    expected = referenced_genes(db)
    have = set(db["geneInfo"]["Symbol"])
    missing = expected - have
    if missing:
        raise AssertionError(
            f"geneInfo is missing {len(missing)} referenced gene(s) {sorted(missing)[:5]}… — "
            "CellChat would silently drop the pairs that use them (README_cellchat.md §4.1).")
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for key, fname in (("interaction", "iscc_interaction_input.csv"),
                       ("complex", "iscc_complex_input.csv"),
                       ("cofactor", "iscc_cofactor_input.csv"),
                       ("geneInfo", "iscc_geneInfo_input.csv")):
        p = os.path.join(out_dir, fname)
        # interaction keeps its interaction_name index (rownames); the others are column-only.
        db[key].to_csv(p, index=(key == "interaction"))
        paths[key] = p
    return paths, expected


def clone_correlation(cell_data, pairs, clone_key="cell_type", exp_key="cell_exp"):
    """Per-candidate-pair clone-correlation — W4's MEASURED analysis axis (DESIGN_cci_spatial.md W0).

    For each pair, how much of its ligand / receptor expression variance is explained by clone
    identity, as the correlation ratio ``eta = sqrt(SS_between / SS_total)`` in ``[0, 1]`` (0 =
    expression independent of clone, 1 = fully determined by clone). Pairs drawn at random land on
    clone-varying copy-number segments by chance, so their expression correlates between neighbouring
    cells purely because neighbours share a clone — an EMERGENT confound, measured here, not planted.

    Parameters
    ----------
    cell_data : dict
        A tumour's ``cell_data`` (needs ``cell_exp`` and a clone label frame).
    pairs : array-like, shape (n_pairs, 2)
        Gene indices ``(ligand, receptor)`` per pair — ``tumor.microenv_truth['cci_pairs']``.
    clone_key : str
        The per-cell clone-label frame; ``"cell_type"`` holds the genotype id (the clone).
    exp_key : str
        The expression frame to correlate; ``"cell_exp"`` (total) by default.

    Returns
    -------
    DataFrame
        One row per pair: ``ligand_gene``, ``receptor_gene``, ``eta_ligand``, ``eta_receptor`` and
        ``eta_max``. Row order matches ``pairs`` (so row 0 is the wired pair).
    """
    exp = cell_data[exp_key]
    X = np.asarray(exp.values, dtype=float)
    clone = np.asarray(cell_data[clone_key].iloc[:, 0].values)
    # group index per cell (dense integer labels for the correlation-ratio sums)
    _, inv = np.unique(clone, return_inverse=True)
    n_groups = inv.max() + 1 if inv.size else 0
    counts = np.bincount(inv, minlength=n_groups).astype(float)

    def eta(gcol):
        x = X[:, gcol]
        if x.size == 0 or np.allclose(x, x[0]):
            return 0.0
        grand = x.mean()
        gsum = np.bincount(inv, weights=x, minlength=n_groups)
        gmean = np.divide(gsum, counts, out=np.zeros_like(gsum), where=counts > 0)
        ss_between = float((counts * (gmean - grand) ** 2).sum())
        ss_total = float(((x - grand) ** 2).sum())
        return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0

    pairs = np.asarray(pairs, dtype=int)
    gene_names = list(exp.columns)
    out = []
    for lig_i, rec_i in pairs:
        el, er = eta(int(lig_i)), eta(int(rec_i))
        out.append({"ligand_gene": gene_names[int(lig_i)], "receptor_gene": gene_names[int(rec_i)],
                    "eta_ligand": el, "eta_receptor": er, "eta_max": max(el, er)})
    return pd.DataFrame(out)
