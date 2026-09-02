"""Q1 clonal sweeps, Q2 diversity over time, Q3 r/K demography — all read `tumor.traces`."""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_tumor, write_json

from iscc.cnevo import diversity_trajectory, growth_phase, sweep_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tumor", required=True)
    p.add_argument("--out-sweep", required=True)
    p.add_argument("--out-diversity", required=True)
    p.add_argument("--out-growth", required=True)
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()

    t = load_tumor(a.tumor)
    write_json(sweep_metrics(t), a.out_sweep)                       # Q1
    diversity_trajectory(t, stride=a.stride).to_csv(a.out_diversity, sep="\t", index=False)  # Q2
    write_json(growth_phase(t), a.out_growth)                       # Q3
    print("dynamics written")


if __name__ == "__main__":
    main()
