"""Build the table of scRNA batch settings fitted to a real atlas (``iscc.data.atlas_scrna_batches``).

Source: the fits of the paper's batch model (``validation/paper/batch_fits.py fit``), which fitted every
tumour of the breast cancer atlas of Wu et al. 2021 (Nat Genet 53:1334-1347,
doi:10.1038/s41588-021-00911-1; GEO GSE176078), each tumour its own 10x run:

* ``estimate_rna`` (``protocol="10x"``, ``fit_dropout=True``) on the tumour's T cells only, so that the
  composition of cell types does not enter the fit: ``mu_lib``, ``sigma_lib``, ``dispersion``,
  ``dropout_mid`` and ``dropout_shape``;
* ``doublet_rate``: 10x's multiplet rate at the tumour's number of cells (Chromium Single Cell 3'
  Reagent Kits v2 User Guide, CG00052 Rev B);
* ``stress_strength``: the dissociation stress, ``max(exp(score - median score) - 1, 0)``, where the
  score is the mean log-normalised expression of the dissociation-response genes of van den Brink et
  al. 2017 in the tumour's T cells.

The ambient fraction is not identifiable from the counts of most tumours, so the table carries none;
``atlas_scrna_batches`` draws it per batch from the registered range.

Output: ``src/iscc/data/reference/wu2021_scrna_batches.tsv``, one row per tumour.

Usage::

    python validation/data/build_atlas_batches.py --fits <batch_fits.py's fits.json>
"""
import argparse
import json
import os

import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "iscc", "data", "reference",
                   "wu2021_scrna_batches.tsv")
COLUMNS = ["mu_lib", "sigma_lib", "dispersion", "dropout_mid", "dropout_shape", "doublet_rate",
           "stress_strength"]


def build(fits_path, out=OUT):
    with open(fits_path) as f:
        fits = json.load(f)
    wu = fits["wu"]
    rows = []
    for t in wu["fitted_tumours"]:
        v = wu["vectors"][t]
        rows.append(dict(tumor=t, n_cells=int(wu["per_tumour"][t]["n_cells"]),
                         **{c: (0.0 if v[c] is None else float(v[c])) for c in COLUMNS}))
    table = pd.DataFrame(rows).set_index("tumor")
    header = (f"# scRNA batch settings fitted to the tumours of Wu et al. 2021 (GEO GSE176078), one 10x run per\n"
              f"# tumour; built by validation/data/build_atlas_batches.py from the fits of {fits['created']}.\n")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(header)
        table.to_csv(f, sep="\t", float_format="%.6g")
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fits", required=True, help="the fits.json that batch_fits.py fit writes")
    table = build(p.parse_args().fits)
    print(f"{len(table)} tumours -> {os.path.normpath(OUT)}")
