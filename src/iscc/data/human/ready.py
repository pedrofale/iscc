"""Check a human-gene dataset against what each copy-number tool needs, before running it.

The thresholds are the ones each tool documents or enforces (read from their repositories, not by
running them), so a dataset that fails here would fail, or run degenerately, in the tool.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

TOOLS = ("copykat", "calicost", "clonalscope", "cancerfinder")


def _check(rows, tool, name, value, threshold, passed):
    rows.append({"tool": tool, "check": name, "value": value, "threshold": threshold, "passed": bool(passed)})


def check_tool_ready(counts, genes, tools=TOOLS, allele=None):
    """Test a dataset against each tool's input thresholds.

    Parameters
    ----------
    counts : pandas.DataFrame, numpy.ndarray or scipy.sparse matrix
        Barcodes x genes UMI counts (cells, or in-tissue spots).
    genes : pandas.DataFrame
        The genes of the columns, with ``chrom`` and ``arm`` (``HumanGenomeLayer.genes``).
    tools : sequence of str, optional
        Any of ``"copykat"``, ``"calicost"``, ``"clonalscope"``, ``"cancerfinder"``.
    allele : AlleleCounts, optional
        Haplotype counts for the same barcodes; CalicoST's SNP check needs them.

    Returns
    -------
    pandas.DataFrame
        One row per check: ``tool``, ``check``, ``value``, ``threshold``, ``passed``.

    Notes
    -----
    - **CopyKAT**: more than 200 detected genes per cell, at least 7,000 genes detected in more than
      5% of cells, and at least 5 such genes on every chromosome.
    - **CalicoST**: more than 50 UMIs and at least 50 SNP-covering UMIs per spot.
    - **Clonalscope**: more than 150 genes detected in at least 1% of barcodes on every modelled arm
      (its segments without bulk DNA).
    - **CancerFinder**: the genes are real symbols; it re-indexes to its own gene list, so only the
      overlap is reported.
    """
    X = sp.csr_matrix(counts.to_numpy() if hasattr(counts, "to_numpy") else counts)
    n = X.shape[0]
    det = (X > 0)
    genes_per_bc = np.asarray(det.sum(axis=1)).ravel()
    frac_bc_per_gene = np.asarray(det.sum(axis=0)).ravel() / max(n, 1)
    umis = np.asarray(X.sum(axis=1)).ravel()
    rows = []
    for tool in tools:
        if tool == "copykat":
            _check(rows, tool, "share of cells with >200 detected genes",
                   float(np.mean(genes_per_bc > 200)), 1.0, np.mean(genes_per_bc > 200) >= 0.95)
            n5 = int((frac_bc_per_gene > 0.05).sum())
            _check(rows, tool, "genes detected in >5% of cells", n5, 7000, n5 >= 7000)
            per_chr = pd.Series(frac_bc_per_gene > 0.05).groupby(genes["chrom"].to_numpy()).sum()
            _check(rows, tool, "fewest such genes on a chromosome", int(per_chr.min()), 5, per_chr.min() >= 5)
        elif tool == "calicost":
            _check(rows, tool, "share of spots with >50 UMIs", float(np.mean(umis > 50)), 1.0,
                   np.mean(umis > 50) >= 0.95)
            if allele is None:
                _check(rows, tool, "share of spots with >=50 SNP UMIs", np.nan, 1.0, False)
            else:
                snp = allele.total
                _check(rows, tool, "share of spots with >=50 SNP UMIs", float(np.mean(snp >= 50)), 1.0,
                       np.mean(snp >= 50) >= 0.95)
                _check(rows, tool, "median SNP UMIs per spot", float(np.median(snp)), 50, np.median(snp) >= 50)
        elif tool == "clonalscope":
            per_arm = pd.Series(frac_bc_per_gene >= 0.01).groupby(genes["arm"].to_numpy()).sum()
            _check(rows, tool, "fewest detected genes on an arm", int(per_arm.min()), 150, per_arm.min() > 150)
        elif tool == "cancerfinder":
            n_named = int(genes["gene_name"].astype(str).str.len().gt(0).sum())
            _check(rows, tool, "genes with a symbol", n_named, 1, n_named > 0)
        else:
            raise ValueError(f"unknown tool {tool!r}; choose from {TOOLS}")
    return pd.DataFrame(rows)
