"""Sample clones and export the copy-number ground truth for them.

Writes the SISTEM-comparable artefacts: the true clone tree in Newick, observed (sampled-clone) and
ancestral copy-number tables, and the derived CNA event log. Ancestral CN here is the TRUE profile
of each ancestor read off the genotype registry, not a parsimony reconstruction.
"""
import argparse, json, os, sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_tumor, write_json

from iscc.cnevo import (
    clone_segment_cn, cna_event_table, segment_coordinates, select_clones, to_medicc2_input,
    true_clone_tree,
)
from iscc.integrations.lineage import LineageTree


def _cn_table(clone_ids, total, allele, coords):
    rows = []
    co = coords.sort_values("segment")
    for i, gid in enumerate(clone_ids):
        for _, c in co.iterrows():
            s = int(c["segment"])
            hap = ("%d,%d" % (allele[i, s, 0], allele[i, s, 1])
                   if allele is not None and pd.notna(allele[i, s, 0]) else "")
            rows.append((gid, c["chrom"], int(c["start"]), int(c["end"]), hap, total[i, s]))
    return pd.DataFrame(rows, columns=["clone", "chrom", "start", "end", "hap_cn", "total_cn"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tumor", required=True)
    p.add_argument("--n-clones", type=int, required=True)
    p.add_argument("--strategy", default="largest")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--medicc2", action="store_true", help="also write a MEDICC2 input TSV")
    a = p.parse_args()

    t = load_tumor(a.tumor)
    os.makedirs(a.out_dir, exist_ok=True)
    sel = select_clones(t, a.n_clones, strategy=a.strategy)
    clones = sel["clones"]
    coords = segment_coordinates(t)
    coords.to_csv(os.path.join(a.out_dir, "segments.tsv"), sep="\t", index=False)
    write_json(sel, os.path.join(a.out_dir, "clone_names.json"))

    ids, total, allele = clone_segment_cn(t, clones)
    _cn_table(ids, total, allele, coords).to_csv(
        os.path.join(a.out_dir, "observed_CN.tsv"), sep="\t", index=False)

    children, root, leaf_of = true_clone_tree(t, clones)
    # every ancestor on the sampled clones' tree, with its TRUE copy number
    anc = sorted({n for n in children} | {c for ks in children.values() for c in ks})
    anc = [n for n in anc if not str(n).endswith("#s") and n != root and n in t.genotypes]
    if anc:
        aid, atot, aall = clone_segment_cn(t, anc)
        _cn_table(aid, atot, aall, coords).to_csv(
            os.path.join(a.out_dir, "ancestral_CN.tsv"), sep="\t", index=False)

    cna_event_table(t, clones).to_csv(
        os.path.join(a.out_dir, "CNA_events.tsv"), sep="\t", index=True, index_label="event_id")

    nwk = LineageTree(t.genotypes_parents, [c for c in clones if c in t.genotypes]).newick()
    with open(os.path.join(a.out_dir, "clone_tree.nwk"), "w") as f:
        f.write(nwk + "\n")

    if a.medicc2:
        to_medicc2_input(t, clones, os.path.join(a.out_dir, "medicc2_input.tsv"), coords=coords)
    print(f"exported {sel['n_clones']} clones (under_sampled={sel['under_sampled']})")


if __name__ == "__main__":
    main()
