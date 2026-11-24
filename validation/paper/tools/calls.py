"""The common output of the copy-number tool runners in this directory, so Fig 5 and P2 score every
tool the same way.

Each runner reads a dataset written by :func:`iscc.data.human.write_visium` or
:func:`iscc.data.human.write_scrna` and writes, in its output directory:

    calls.tsv    barcode, malignant (1, 0, or empty when the tool makes no call), p_malignant (empty
                 when the tool gives no probability), clone (the tool's label, empty when it gives none)
    cn_arm.tsv   barcode, arm, value, scale: the tool's copy-number estimate per chromosome arm, on
                 the tool's own scale ("total_cn", "log2ratio" or "relative")
    run.json     tool, version, settings and wall time
    native/      the tool's own output files

The R runners (copykat.R, clonalscope.R) write the same columns.
"""
import json
import os

import numpy as np
import pandas as pd

CALL_COLUMNS = ["barcode", "malignant", "p_malignant", "clone"]


def arms(dataset_dir):
    """The dataset's arm table: ``arm``, ``chr`` (integer), ``start``, ``end``."""
    return pd.read_csv(os.path.join(dataset_dir, "clonalscope", "chrarm.tsv"), sep="\t")


def arm_of(chrom, pos, arm_table):
    """The arm (e.g. ``"7q"``) of each (chromosome, position); empty outside the modelled arms."""
    chrom = pd.Series(chrom).astype(str).str.replace("chr", "", regex=False).to_numpy()
    pos = np.asarray(pos)
    out = np.full(len(pos), "", dtype=object)
    for r in arm_table.itertuples():
        hit = (chrom == str(r.chr)) & (pos >= r.start) & (pos < r.end)
        out[hit] = r.arm
    return out


def write(out_dir, calls, cn_arm=None, **run):
    """Write ``calls.tsv`` (and ``cn_arm.tsv`` if given) and ``run.json`` into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    calls = calls.reindex(columns=CALL_COLUMNS)
    calls.to_csv(os.path.join(out_dir, "calls.tsv"), sep="\t", index=False, na_rep="")
    if cn_arm is not None:
        cn_arm.reindex(columns=["barcode", "arm", "value", "scale"]).to_csv(
            os.path.join(out_dir, "cn_arm.tsv"), sep="\t", index=False)
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=1, default=str)
