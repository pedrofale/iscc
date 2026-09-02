"""Q4 — the copy-number landscape and how it evolves."""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_tumor, write_json

from iscc.cnevo import cn_landscape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tumor", required=True)
    p.add_argument("--out-trajectory", required=True)
    p.add_argument("--out-summary", required=True)
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()

    traj, summary = cn_landscape(load_tumor(a.tumor), stride=a.stride)
    traj.to_csv(a.out_trajectory, sep="\t", index=False)
    write_json(summary, a.out_summary)
    print("landscape written")


if __name__ == "__main__":
    main()
