"""Q7 — spatial structure and multi-focality.

Writes a null payload for unstructured runs, so the rule graph is the same in both scenarios and
the summary table keeps one schema.
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_tumor, write_json

from iscc.cnevo import is_structured, spatial_structure


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tumor", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    t = load_tumor(a.tumor)
    res = spatial_structure(t)
    write_json({"structured": bool(is_structured(t)), **(res or {})}, a.output)
    print("structure written" if res else "unstructured: null payload written")


if __name__ == "__main__":
    main()
