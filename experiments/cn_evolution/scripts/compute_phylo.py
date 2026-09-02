"""Q5 CN data quality and Q6 tree-reconstruction potential, for one sampled clone set."""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_tumor, write_json

from iscc.cnevo import data_quality, reconstruction_potential


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tumor", required=True)
    p.add_argument("--clone-names", required=True)
    p.add_argument("--out-quality", required=True)
    p.add_argument("--out-reconstruction", required=True)
    a = p.parse_args()

    t = load_tumor(a.tumor)
    with open(a.clone_names) as f:
        sel = json.load(f)
    clones, n_req = sel["clones"], sel.get("n_requested")
    write_json(data_quality(t, clones, n_requested=n_req), a.out_quality)          # Q5
    write_json(reconstruction_potential(t, clones), a.out_reconstruction)          # Q6
    print("phylo written")


if __name__ == "__main__":
    main()
